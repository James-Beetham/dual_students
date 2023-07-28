import argparse,json,os,random,datetime,logging,math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models

import my_utils
import approximate_gradients
from dataloader import get_dataloader
import network


def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def generator_loss(args, s_logit, t_logit,  z = None, z_logit = None, reduction="mean"):
    assert 0 
    
    loss = - F.l1_loss( s_logit, t_logit , reduction=reduction) 
    
            
    return loss


def train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()
    
    optimizer_S,  optimizer_G = optimizer

    gradients = []
    args.train_tqdm.reset()
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            #Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            #Get fake image from generator
            fake = generator(z, pre_x=args.approx_grad) # pre_x returns the output of G before applying the activation


            ## APPOX GRADIENT
            approx_grad_wrt_x, loss_G = approximate_gradients.estimate_gradient_objective(args, teacher, student, fake, 
                                                epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, 
                                                device=device, pre_x=True)

            fake.backward(approx_grad_wrt_x)
                
            optimizer_G.step()

            if i == 0 and args.rec_grad_norm:
                x_true_grad = my_utils.measure_true_grad_norm(args, fake)

        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad(): 
                t_logit = teacher(fake)

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()


            s_logit = student(fake)


            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        # Log Results
        args.train_tqdm.set_description(args.train_tqdm_desc.format(loss_G.item(),loss_S.item()))
        args.train_tqdm.update()
        if i % args.log_interval == 0:
            # args.logger.info(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')
            
            if i == 0:
                with open(args.log_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f,%f\n"%(epoch, loss_G, loss_S))


            if args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
                if i == 0:
                    with open(args.log_dir + "/norm_grad.csv", "a") as f:
                        f.write("%d,%f,%f,%f\n"%(epoch, G_grad_norm, S_grad_norm, x_true_grad))
                    

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            return 


def test(args, student = None, generator = None, device = "cuda", test_loader = None, epoch=0):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    total = 0
    args.test_tqdm.reset()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            args.test_tqdm.set_description(args.test_tqdm_desc.format(test_loss/total,correct/total*100))
            args.test_tqdm.update()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    # args.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     accuracy))
    with open(args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n"%(epoch, accuracy))
    acc = correct/len(test_loader.dataset)
    return acc

def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return  np.mean(G_grad), np.mean(S_grad)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=1, help = "Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help = "Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help = "Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'],)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"],)
    parser.add_argument('--steps', nargs='+', default = [0.1, 0.3, 0.5], type=float, help = "Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help = "Fractional decrease in lr")

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn','cifar10'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet34_8x', choices=my_utils.classifiers, help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=int, default=0, help='Cuda device to use, or -1 to use cpu.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
    

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--log_dir', type=str, default='results')
    parser.add_argument('--log_level', type=str, choices=logging._nameToLevel, default='INFO')
    parser.add_argument('--disable_tqdm', default=False, action='store_true', help='Disables tqdm output.')
    parser.add_argument('--save_config', type=int, default=1, help='Whether to save the config file.')
    parser.add_argument('--config', nargs='+',default=[], help='One or more config files to load arguments from.'+
                        'Note, config values overwrite all values in the order of config files specified.')

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3) 
    

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')
    

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0) 

    parser.add_argument('--store_checkpoints', type=int, default=1)

    parser.add_argument('--student_model', type=str, default='resnet18_8x',
                        help='Student model architecture (default: resnet18_8x)')


    args = parser.parse_args()
    my_utils.setup_args(args)

    args.logger.info(f'torch version: {torch.__version__}')

    args.query_budget *=  10**6
    args.query_budget = int(args.query_budget)
    if args.MAZE:

        args.logger.info("\n"*2)
        args.logger.info("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        args.logger.info("\n"*2)
        args.scheduer = "cosine"
        args.loss = "kl"
        args.batch_size = 128
        args.g_iter = 1
        args.d_iter = 5
        args.grad_m = 10
        args.lr_G = 1e-4 
        args.lr_S = 1e-1


    if args.student_model not in my_utils.classifiers:
        if "wrn" not in args.student_model:
            raise ValueError("Unknown model")

    args.logger.info(args.log_dir)
    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)

    with open(args.log_dir + "/loss.csv", "w") as f:
        f.write("epoch,loss_G,loss_S\n")

    with open(args.log_dir + "/accuracy.csv", "w") as f:
        f.write("epoch,accuracy\n")

    if args.rec_grad_norm:
        with open(args.log_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    with open('latest_experiments.txt', 'a') as f: f.write(f'{datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\t{args.log_dir}\n')

    use_cuda = not args.cuda < 0 and torch.cuda.is_available()
    args.device = torch.device(f'cuda:{args.cuda}' if use_cuda else 'cpu')

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Preparing checkpoints for the best Student
    global file
    model_dir = f"checkpoint/student_{args.model_id}"; args.model_dir = model_dir
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    # with open(f"{model_dir}/model_info.txt", "w") as f:
    #     json.dump(args.__dict__, f, indent=2)  
    file = open(f"{args.model_dir}/logs.txt", "w") 

    # Eigen values and vectors of the covariance matrix
    _, test_loader = get_dataloader(args)


    args.normalization_coefs = None
    args.G_activation = torch.tanh

    num_classes = 10 if args.dataset in ['cifar10', 'svhn'] else 100
    args.num_classes = num_classes

    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
        if args.dataset == 'svhn':
            args.logger.info("Loading SVHN TEACHER")
            args.ckpt = 'checkpoint/teacher/svhn-resnet34_8x.pt'
        teacher.load_state_dict( torch.load( args.ckpt, map_location=args.device) )
    else:
        teacher = my_utils.get_classifier(args.model, pretrained=True, num_classes=args.num_classes)
    
    

    teacher.eval()
    teacher = teacher.to(args.device)
    args.logger.info(f'Teacher restored from: {args.ckpt}') 
    args.logger.info(f'Training with {args.model} as a Target') 
    correct = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader),disable=args.disable_tqdm):
            data, target = data.to(args.device), target.to(args.device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    args.logger.info(f'Teacher - Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)')
        
    student = my_utils.get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes)
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32, activation=args.G_activation)

    student = student.to(args.device)
    generator = generator.to(args.device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    if args.student_load_path :
        # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
        student.load_state_dict( torch.load( args.student_load_path ) )
        args.logger.info(f'Student initialized from {args.student_load_path}')
        acc = test(args, student=student, generator=generator, device = args.device, test_loader = test_loader)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    args.logger.info(f'Total budget: {args.query_budget//1000}k')
    args.logger.info(f'Cost per iterations: {args.cost_per_iteration}')
    args.logger.info(f'Total number of epochs: {number_epochs}')

    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )

    if args.MAZE:
        optimizer_G = optim.SGD( generator.parameters(), lr=args.lr_G , weight_decay=args.weight_decay, momentum=0.9 )    
    else:
        optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
    
    steps = sorted([int(step * number_epochs) for step in args.steps])
    args.logger.info(f'Learning rate scheduling at steps: {steps}')

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)
    schedulers = [scheduler_S,scheduler_G]

    best_acc = 0
    acc_list = []

    epoch_desc = 'Epochs: [test_acc {:.02f}% ({:.02f}% best)]'
    epoch_tqdm = tqdm(range(1,number_epochs + 1), desc=epoch_desc.format(0,0), disable=args.disable_tqdm, position=0)
    args.train_tqdm_desc = 'Train: [loss G={:.04f}|S={:.04f}]'
    args.train_tqdm = tqdm(total=args.epoch_itrs, desc=args.train_tqdm_desc.format(0,0), disable=args.disable_tqdm, position=1)    
    args.test_tqdm_desc = 'Test: [loss S={:.04f}][acc {:.02f}]'
    args.test_tqdm = tqdm(total=math.ceil(len(test_loader.dataset)/args.batch_size), desc=args.test_tqdm_desc.format(0,0), disable=args.disable_tqdm, position=2)
    for epoch in epoch_tqdm:
        # Train
        train(args, teacher=teacher, student=student, generator=generator, device=args.device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        if args.scheduler != "none": [v.step() for v in schedulers]
        # Test
        acc = test(args, student=student, generator=generator, device = args.device, test_loader = test_loader, epoch=epoch)
        epoch_tqdm.set_description(epoch_desc.format(acc*100,best_acc*100))
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            name = 'resnet34_8x'
            torch.save(student.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}.pt")
            torch.save(generator.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}-generator.pt")
        if args.store_checkpoints:
            torch.save(student.state_dict(), args.log_dir + f"/checkpoint/student.pt")
            torch.save(generator.state_dict(), args.log_dir + f"/checkpoint/generator.pt")
    args.logger.info("Best Acc=%.6f"%best_acc)

    with open(args.log_dir + "/Max_accuracy = %f"%best_acc, "w") as f: f.write(" ")

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%s.csv'%(args.dataset), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    main()


