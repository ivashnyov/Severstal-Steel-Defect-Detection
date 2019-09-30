import functools
import torch.utils.model_zoo as model_zoo
import numpy as np
from resnet import resnet_encoders
#from .dpn import dpn_encoders
#from .vgg import vgg_encoders
#from .senet import senet_encoders
#from .densenet import densenet_encoders
#from .inceptionresnetv2 import inception_encoders


encoders = {}
encoders.update(resnet_encoders)
#encoders.update(dpn_encoders)
#encoders.update(vgg_encoders)
#encoders.update(senet_encoders)
#encoders.update(densenet_encoders)
#encoders.update(inception_encoders)




def preprocess_input(x, mean=None, std=None, input_space='RGB', input_range=None, **kwargs):

    if input_space == 'BGR':
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained='imagenet'):
    settings = encoders[encoder_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError('Avaliable pretrained options {}'.format(settings.keys()))
    
    formatted_settings = {}
    formatted_settings['input_space'] = settings[pretrained].get('input_space')
    formatted_settings['input_range'] = settings[pretrained].get('input_range')
    formatted_settings['mean'] = settings[pretrained].get('mean')
    formatted_settings['std'] = settings[pretrained].get('std')
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained='imagenet'):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)