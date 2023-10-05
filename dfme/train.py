import argparse,json,os,random,datetime,logging,math,dataclasses,time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.utils.data as torch_utils_data

import my_utils
import approximate_gradients
from dataloader import get_dataloader
import network


@dataclasses.dataclass(frozen=True,order=True)
class TrainTestOpts():
    student: torch.nn.Module
    generator: torch.nn.Module
    student_loss: torch.nn.Module
    device: torch.device
    use_tqdm: tqdm
    use_tqdm_desc: str
@dataclasses.dataclass(frozen=True,order=True)
class TrainOpts(TrainTestOpts):
    teacher: torch.nn.Module
    generator_loss: torch.nn.Module
    student_optimizer: list[optim.Optimizer]
    generator_optimizer: optim.Optimizer
    args: argparse.Namespace
def train(opts:TrainOpts):
    """Main Loop for one epoch of Training Generator and Student"""
    train_logs = dict()
    train_time = time.time()
    train_generator_duration = 0
    train_student_duration = 0
    tqdm_desc = 'Train: [loss G={:0.4f}|S{:0.4f}]'
    opts.use_tqdm.set_description(opts.use_tqdm_desc.format(tqdm_desc.format(0,0)))
    args = opts.args
    global file
    opts.teacher.eval()
    opts.use_tqdm.reset()
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        opts.generator.train()
        opts.student.eval()
        generator_time = time.time()
        for _ in range(args.g_iter):
            #Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(opts.device)
            opts.generator_optimizer.zero_grad()
            #Get fake image from generator
            fake:torch.Tensor = opts.generator(z, pre_x=args.approx_grad) # pre_x returns the output of G before applying the activation

            if args.num_students == 1:
                ## APPOX GRADIENT
                approx_grad_wrt_x, loss_G = approximate_gradients.estimate_gradient_objective(args, opts.teacher, opts.student, fake, 
                                                    epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, 
                                                    device=opts.device, pre_x=True)
                fake.backward(approx_grad_wrt_x)
            else:
                s_logit:list[torch.Tensor] = opts.student(fake,combine=False)
                # all combinations of i,j (students)
                combs = [(i,j) for i in range(len(s_logit)) for j in range(i+1,len(s_logit))]
                # maximize distance between logits for students (aka students disagree)
                loss_G = opts.generator_loss(s_logit[0],s_logit[1])
                for i,j in combs[1:]: loss_G += opts.generator_loss(s_logit[i],s_logit[j])
                loss_G = -loss_G / len(combs)
                loss_G.backward()
            opts.generator_optimizer.step()

        x_true_grad = None
        if i == 0 and args.rec_grad_norm:
            x_true_grad = my_utils.measure_true_grad_norm(opts, fake)
        train_generator_duration += time.time() - generator_time

        opts.generator.eval()
        opts.student.train()
        student_time = time.time()
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, args.nz)).to(opts.device)
            fake = opts.generator(z).detach()
            for v in opts.student_optimizer: v.zero_grad()

            with torch.no_grad(): t_logit:torch.Tensor = opts.teacher(fake)

            if args.num_students == 1:
                s_logit:list[torch.Tensor] = opts.student(fake,combine=False)

                loss_S = torch.stack([opts.student_loss(v,t_logit.detach()) for v in s_logit]).sum() 
                loss_S:torch.Tensor = loss_S / args.num_students
                loss_S.backward()
                for v in opts.student_optimizer: v.step() # there should only be 1
            elif isinstance(opts.student,network.multi_student.MultiStudentModel):
                for m,o in zip(opts.student.models,opts.student_optimizer):
                    s_logit:list[torch.Tensor] = m(fake)
                    loss_S = opts.student_loss(s_logit,t_logit.detach()).sum() 
                    loss_S.backward()
                    o.step()
            else: raise ValueError(f'Unknown student model type',type(opts.student))
        train_student_duration += time.time() - student_time

        # Log Results
        opts.use_tqdm.set_description(opts.use_tqdm_desc.format(tqdm_desc.format(loss_G.item(),loss_S.item())))
        opts.use_tqdm.update()
        if i % args.log_interval == 0: 
            # args.logger.info(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')
            pass
        if i == 0:
            with open(args.log_dir + "/loss.json.txt", "a") as f:
                f.write(json.dumps(dict(epoch=args.epoch,
                                        generator_loss=loss_G.item(),
                                        student_loss=loss_S.item()))+'\n')
        if x_true_grad != None:
            G_grad_norm, S_grad_norm = compute_grad_norms(opts.generator, opts.student)
            if i == 0:
                with open(args.log_dir + "/norm_grad.json.txt", "a") as f:
                    f.write(json.dumps(dict(epoch=args.epoch,
                                            generator_grad=G_grad_norm.item(),
                                            student_grad=S_grad_norm.item(),
                                            true_grad=None if x_true_grad == None else x_true_grad.item()))+'\n')

        # update query budget
        args.query_budget -= args.cost_per_iteration
        if args.query_budget < args.cost_per_iteration:
            break
    train_time = time.time() - train_time
    train_logs['time_p'] = my_utils.pretty_time(train_time)
    train_logs['time_p_generator'] = my_utils.pretty_time(train_generator_duration)
    train_logs['time_p_student'] = my_utils.pretty_time(train_student_duration)
    train_logs['time_sec'] = train_time
    train_logs['time_sec_generator'] = train_generator_duration
    train_logs['time_sec_student'] = train_student_duration
    return train_logs

@dataclasses.dataclass(frozen=True,order=True)
class TestOpts(TrainTestOpts):
    loader: torch_utils_data.DataLoader
def test(opts:TestOpts):
    global file
    opts.student.eval()

    tqdm_desc = 'Test: [loss S={:0.4f}][acc {:0.2f}]'
    opts.use_tqdm.set_description(opts.use_tqdm_desc.format(tqdm_desc.format(0,0)))
    loss_sum = 0
    correct = 0
    total = 0
    opts.use_tqdm.reset()
    with torch.no_grad():
        for i, (data, target) in enumerate(opts.loader):
            data: torch.Tensor; target: torch.Tensor
            data, target = data.to(opts.device), target.to(opts.device)
            output:torch.Tensor = opts.student(data)

            loss_sum += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            opts.use_tqdm.set_description(opts.use_tqdm_desc.format(
                tqdm_desc.format(loss_sum/total,correct/total*100)))
            opts.use_tqdm.update()

    loss_sum /= len(opts.loader.dataset)
    accuracy = correct / len(opts.loader.dataset)
    ret_log = dict(
        loss=loss_sum,
        accuracy=accuracy
    )
    return accuracy, ret_log

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
    parser.add_argument('--skip_test_model', default=False, action='store_true', help='Skips evaluating teacher first.')
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
    parser.add_argument('--num_students', type=int, default=1,
                        help='Number of students to use.')
    parser.add_argument('--combine_student_outputs', type=str, default='first', 
                        choices=list(network.multi_student.COMBINE_MODES.keys()),
                        help='How to get single model output for multiple models (for testing).')

    args = parser.parse_args()
    my_utils.setup_args(args)

    start_time = time.time()
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


    args.logger.info(args.log_dir)
    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)

    with open('latest_experiments.txt', 'a') as f: 
        f.write(f'{datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\t{args.log_dir}\n')

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

    student, teacher = my_utils.build_student_teacher(args)
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32, activation=args.G_activation)
    [v.to(args.device) for v in [student,teacher,generator]]

    student_loss,generator_loss = my_utils.build_criterion(args)

    teacher.eval()
    args.logger.info(f'Teacher restored from: {args.ckpt}') 
    if not args.skip_test_model:
        args.logger.info(f'Training with {args.model} as a Target') 
        teacher_acc,teacher_log = test(TestOpts(student=teacher,generator=None,student_loss=student_loss,
                    device=args.device,
                    use_tqdm=tqdm(disable=args.disable_tqdm,total=math.ceil(len(test_loader.dataset)/args.batch_size)),
                    use_tqdm_desc='Teacher {}',
                    loader=test_loader))
        args.logger.info(f'Teacher - Test set: Accuracy: {teacher_acc*100:.2f}% (/{len(test_loader.dataset)})')

    if args.student_load_path:
        # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
        student.load_state_dict( torch.load( args.student_load_path ) )
        student_tqdm_desc = 'Student Test: [loss S={:.04f}][acc {:.02f}]'
        acc,student_start_log = test(TestOpts(student=student,generator=generator,
                student_loss=student_loss,
                device=args.device,
                use_tqdm=tqdm(disable=args.disable_tqdm,total=math.ceil(len(test_loader.dataset)/args.batch_size)),
                use_tqdm_desc='Student ',
                loader=test_loader))
        args.logger.info(f'Student initialized from {args.student_load_path} ({acc*100:.2f}%)')

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.d_iter)
    number_epochs = args.number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    args.logger.info(f'Total budget: {args.query_budget//1000}k')
    args.logger.info(f'Cost per iterations: {args.cost_per_iteration}')
    args.logger.info(f'Total number of epochs: {number_epochs}')

    optimizer_S, optimizer_G, schedulers = my_utils.build_optimizers_and_schedulers(args,student,generator)

    best_acc = 0
    acc_list = []

    epoch_desc = 'Epochs: [test_acc {:.02f}% ({:.02f}% best)]'
    epoch_tqdm = tqdm(range(1,number_epochs + 1), desc=epoch_desc.format(0,0), disable=args.disable_tqdm, position=0)
    trainOpts = TrainOpts(
        student=student,teacher=teacher,generator=generator,
        student_loss=student_loss,generator_loss=generator_loss,
        student_optimizer=optimizer_S,generator_optimizer=optimizer_G,
        device=args.device,args=args,
        use_tqdm=tqdm(total=args.epoch_itrs, disable=args.disable_tqdm, position=1),
        use_tqdm_desc='{}')
    testOpts = TestOpts(
        student=student,generator=generator,
        student_loss=student_loss,
        device=args.device,
        loader=test_loader,
        use_tqdm=tqdm(total=math.ceil(len(test_loader.dataset)/args.batch_size), 
                      disable=args.disable_tqdm, position=2),
        use_tqdm_desc='{}')
    for epoch in epoch_tqdm:
        # Train
        args.epoch = epoch
        train_logs = train(trainOpts)
        if args.scheduler != "none": [v.step() for v in schedulers]
        # Test
        acc,test_logs = test(testOpts)
        epoch_tqdm.set_description(epoch_desc.format(acc*100,best_acc*100))
        with open(os.path.join(args.log_dir,'logs.json.txt'),'a') as f:
            f.write(json.dumps({**dict(
                                    epoch=epoch,
                                    time_remaining=my_utils.pretty_time((time.time()-start_time)/epoch*number_epochs),
                                    time_taken=my_utils.pretty_time(time.time()-start_time),
                                    time=time.time()),
                                **train_logs,
                                **test_logs})+'\n')
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

    with open(args.log_dir + f'/Max_accuracy = {best_acc*100:0.4f}', 'w') as f: f.write(' ')
    args.logger.info(f'Finished ({best_acc*100:02f}%) in {my_utils.pretty_time(time.time()-start_time)}')

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%s.csv'%(args.dataset), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    main()


