import segmentation_models_pytorch as smp
import yaml
from models import MSFormer_v1, MSFormer_v2, MSFormer_v3, MSFormer_DAT_AGF, MSFormer_SAE_AGF
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights
from deepcrack import DeepCrack
from crackformer import CrackFormer


SMP_MODELS = [
    'Unet',
    'UnetPlusPlus',
    'PSPNet',
    'PAN',
    'MAnet',
    'Linknet',
    'FPN',
    'DeepLabV3',
    'DeepLabV3Plus',
    'UPerNet',
    'Segformer'
]

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


### Create models for the different baselines using segmentation models pytorch
def create_smp_model(architecture='Unet', encoder_name='resnet50', in_channels=3, 
                     num_classes=1, encoder_weights='imagenet'):
    model = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model # .to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def define_model(architecture, num_classes=1):
    if architecture in SMP_MODELS:
        model = create_smp_model(architecture=architecture, num_classes=num_classes)
    elif architecture == 'MSFormer_baseline':
        model = create_smp_model(architecture='Unet', encoder_name='resnet18', 
                                 in_channels=3, num_classes=num_classes)
    elif architecture == 'MSFormer_v1':
        model = MSFormer_v1(input_dim=3, embed_size=config['model']['nChannel'], 
                            img_size=config['data']['image_size'], large_nlayers=[1], 
                            small_nlayers=[1], fine_nlayers=[1])
    elif architecture == 'MSFormer_v2':
        model = MSFormer_v2(input_dim=3, embed_size=config['model']['nChannel'], 
                            img_size=config['data']['image_size'], large_nlayers=[1], 
                            small_nlayers=[1], fine_nlayers=[1])
    elif architecture == 'MSFormer_v3':
        model = MSFormer_v3(input_dim=3, embed_size=config['model']['nChannel'], 
                            img_size=config['data']['image_size'], large_nlayers=[1], 
                            small_nlayers=[1], fine_nlayers=[1])
    elif architecture == 'MSFormer_SAE_AGF':
        model = MSFormer_SAE_AGF(input_dim=3, embed_size=config['model']['nChannel'], 
                                 img_size=config['data']['image_size'])
    elif architecture == 'MSFormer_DAT_AGF':
        model = MSFormer_DAT_AGF(input_dim=3, embed_size=config['model']['nChannel'], 
                                 img_size=config['data']['image_size'])
    elif architecture == 'FCN':
        model = fcn_resnet50(num_classes=1, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    elif architecture == 'DeepCrack':
        model = DeepCrack(num_classes=num_classes)
    elif architecture == 'CrackFormer':
        model = CrackFormer()
    return model