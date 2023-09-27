import logging,os,argparse,string,shutil,json,sys,typing

from cifar10_models import *
from approximate_gradients import *
import network
import train

def setup_args(args:argparse.Namespace, logger_name='default'):
    """Checks and sets up common default arguments.

    Optional attributes:
    - (log_level) Create and add logger to namespace.
    - (dotenv_file) Load dotenv file.
    - (data_dir) Load data_dir from dotenv_file if not present.
    - (config) One or more paths to config files, overwrites args in order.
    - (log_dir) Directory to save experiment details.
        - (overwrite) Will proactively remove existing directory if it exists
                      or will try adding random extensions to the log_dir to not
                      overwrite existing directories.
        - (save_config) Whether to save the arguments for this run in a config file.
        - (log_file) Sets if log_file is not specified.
    Args:
        args (argparse.Namespace): Namespace to work with.
        logger_name (str, optional): Name of the logger. Defaults to 'default'.

    Raises:
        ValueError: If log_level is invalid.
    """
    if hasattr(args,'dotenv_file'):
        import dotenv
        defualt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'.env')
        if args.dotenv_file == None\
                or len(args.dotenv_file) == 0\
                or (args.dotenv_file=='.env' and not os.path.isfile(args.dotenv_file)):
            args.dotenv_file = defualt_file
        assert os.path.isfile(args.dotenv_file), f'Invalid .env file:\n\t{args.dotenv_file}'
        assert dotenv.load_dotenv(args.dotenv_file),\
                f'Could not find .env file:\n\t{os.path.abspath(args.dotenv_file)}'

    if hasattr(args,'config') and args.config != None:
        if type(args.config) == str: args.config = [args.config]
        assert type(args.config) == list, f'Invalid config type: {type(args.config)}.'
        for config_file in args.config:
            assert os.path.isfile(config_file), f'Config file does not exist.\n\tFile: {config_file}'
            with open(config_file,'r') as f: cfg = json.load(f)
            for key,val in cfg.items(): setattr(args,key,val)

    if hasattr(args,'data_dir'):
        if args.data_dir == None:
            assert hasattr(args,'dotenv_file'), f'--dotenv_file was not specified'
            assert 'DATA_FOLDER' in os.environ, f'DATA_FOLDER was not in .env'
            args.data_dir = os.getenv('DATA_FOLDER')
        assert os.path.isdir(args.data_dir), f'Invalid data_dir:\n\t{args.data_dir}'
    
    if hasattr(args,'log_dir'):
        if hasattr(args,'overwrite'):
            if os.path.isdir(args.log_dir):
                if not args.overwrite:
                    rand = random.Random(17); chars = string.digits+string.ascii_lowercase
                    log_dirs = [args.log_dir+'-'+''.join([rand.choice(chars) for j in range(4)]) for i in range(100)]
                    for log_dir in log_dirs: 
                        if not os.path.isdir(log_dir): break
                    if os.path.isdir(log_dir):
                        raise FileExistsError(f'Directory log_dir exists and an unused extension could not be found.\n\tPath: {args.log_dir}')
                else:
                    shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)

        if not hasattr(args, 'log_file') or args.log_file == None or len(args.log_file) == 0:
            args.log_file = os.path.join(args.log_dir,'logs.txt')

        if hasattr(args, 'save_config') and args.save_config == 1:
            cfg = {k:v for k,v in vars(args).items() if is_jsonable(v)}
            with open(os.path.join(args.log_dir,'config.json'),'w') as f: json.dump(cfg,f)

    if hasattr(args,'log_level'):
        args.log_level = args.log_level.upper()
        if args.log_level in logging._nameToLevel: args.log_level = logging._nameToLevel[args.log_level]
        elif args.log_level.isdigit(): args.log_level = int(args.log_level) 
        else: raise ValueError('log_level', args.log_level, logging._nameToLevel)
        log_file = args.log_file if hasattr(args, 'log_file') else None
        args.logger = custom_logger(logger_name,
                                    level=args.log_level,file_path=log_file)

    

def logger_enabled_for(logger_or_args, level)->bool:
    """Check whether logger is enabled for level.

    Args:
        logger_or_args (argparse.Namespace or logging.Logger): Logger (or namespace.logger) to check.
        level (str or number): Level to check (logging.name string or number)

    Returns:
        bool: True if enabled, else False.
    """
    logger = logger_or_args
    if type(logger_or_args) == argparse.Namespace:
        assert hasattr(logger_or_args,'logger'), f'logger attribute not present'
        logger = logger_or_args.logger
    assert type(logger) == logging.Logger, f'Invalid logger type: {type(logger_or_args)}'
    lv = level
    if type(level) == str:
        level = level.upper()
        assert level in logging._nameToLevel, f'Invalid level string: {level}\n\tValid: {",".join([k for k in logging._nameToLevel])}'
        lv = logging._nameToLevel[level]
    return logger.isEnabledFor(lv)

def logger_disabled_for(*args,**kwargs):
    """See logger_enabled_for(...)."""
    return not logger_enabled_for(*args,**kwargs)

# https://stackoverflow.com/questions/28330317/print-timestamp-for-logging-in-python
def custom_logger(name,level=30,file_path=None,print_out=True):
    formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)-8s\t%(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if file_path != None:
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if print_out:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
    return logger

def pretty_time(secs):
    m,s = divmod(secs,60); h,m = divmod(m,60); d,h = divmod(h,24)
    dhms = [d>0]
    for v in [h,m,s]: dhms.append(dhms[-1] or v>0)
    ret = ''
    if dhms[0]: ret+=f'{d}-'
    if dhms[1]: ret+=f'{h:02.0f}:'
    if dhms[2]: ret+=f'{m:02.0f}:'
    if dhms[3]: ret+=f'{s:05.2f}s'
    return ret

def log_done(args, start_time:float,force_print=False,prepend='',append='')->str:
    if type(args) == logging.Logger: logger = args
    elif type(args) == argparse.Namespace: logger = args.logger
    done_str = prepend+f'Done in {pretty_time(time.time()-start_time)}'+append
    logger.info(done_str)
    if force_print and args.logger.level > logging.INFO: print(done_str)
    return done_str

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_classifier(classifier, pretrained=True, num_classes=10):
    if classifier == "wrn-28-10":
        net =  wrn(
                    num_classes=num_classes,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3
                )
        if pretrained:
            state_dict = torch.load("cifar100_models/state_dicts/model_best.pt", map_location=device)["state_dict"]
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        return net
    elif 'wrn' in classifier and 'kt' not in classifier:
        depth = int(classifier.split("-")[1])
        width = int(classifier.split("-")[2])

        net =  wrn(
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=width
                )
        if pretrained:
            raise ValueError("Cannot be pretrained")
        return net
    elif classifier == "kt-wrn-40-2":
        net = WideResNetKT(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        if pretrained:
            state_dict = torch.load("cifar10_models/state_dicts/kt_wrn.pt", map_location=device)["state_dict"]
            net.load_state_dict(state_dict)
        return net
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        return network.resnet_8x.ResNet18_8x(num_classes=num_classes)

    else:
        raise NameError('Please enter a valid classifier')


def measure_true_grad_norm(opts:train.TrainOpts, x:torch.Tensor):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(opts.args, opts.teacher, opts.student, x, pre_x=True, device=opts.device)
    if true_grad == None: return None
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad:torch.Tensor = true_grad.norm(2, dim=1).mean().cpu()
    return norm_grad

classifiers = [
    "resnet34_8x", # Default DFAD
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "mobilenet_v2",
    "googlenet",
    "inception_v3",
    "wrn-28-10",
    "resnet18_8x",
    "kt-wrn-40-2",
]