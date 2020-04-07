import yaml

def load_yaml(load_path):
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded
    
def get_ckpt_inf(ckpt_path, steps_per_epoch):
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs
    return epochs, steps+1