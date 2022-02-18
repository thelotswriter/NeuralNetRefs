import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import copy
import tkinter as tk
from tkinter import filedialog


# Provides a GUI (through tkinter) to get access to image paths
def get_image_path():
    root = tk.Tk()
    # Remove root window
    root.withdraw()
    # Run the dialog. Only .jpg files will work, so we restrict to those
    return filedialog.askopenfilename(filetypes=[("JPEG", ".jpg")])


# Load an image and transform it to the given size square.
# Returns the image as a tensor on the appropriate device
def load_image(imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    # Get the image
    image = Image.open(get_image_path())
    # Resize the image and make it a torch tensor
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# A class for normalizing images for training
class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # Create mean and standard deviation tensors
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Return a normalized image
        return (img - self.mean) / self.std


# Show a tensor as an image
def imshow(tensor, title=None):
    # Tensor to image
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    # Remove empty channel 0
    image = image.squeeze(0)
    image = unloader(image)
    # Show the image with title, if one is given
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# Class for calculating the content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # The target is a stated value, not a variable. Therefore it is detached
        self.target = target.detach()

    def forward(self, input):
        # Calculate mean squared error loss
        self.loss = F.mse_loss(input, self.target)
        return input


# Function for computing the Gram matrix. Basically dot product
def gram_matrix(input):
    batch_size, num_feat_maps, r, c = input.size()
    # Shape features into a matrix
    features = input.view(batch_size * num_feat_maps, r * c)
    # Calculate Gram Matrix (Feature matrix * transpose)
    G = torch.mm(features, features.t())
    # Return Gram Matrix after normalizing
    return G.div(batch_size * num_feat_maps * r * c)


# Class for calculating the style loss
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        # As with content loss, target isn't a variable. Detach the Gram Matrix
        # For style, we use the Gram matrix
        self.target = gram_matrix(target).detach()

    # As with content loss, but with Gram matrices!
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# Obtains the model (layers from vgg19) and attached losses
def get_style_model_and_losses(vgg, content_img, style_img, device):
    # Layers with features related to content and style
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    normalization = Normalization(device).to(device)
    # Arrays for the two losses
    content_losses = []
    style_losses = []
    # Our model which will take layers from vgg
    model = nn.Sequential(normalization)
    # Variable i keeps track of where we are in the original net
    i = 0
    for layer in vgg.children():
        # Increment the iterator on convolutions and add set the name
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        # ReLU (activiation) layers
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        # Pooling layer
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        # BatchNorm layer
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        # Shouldn't be any other kinds of layers
        else:
            raise RuntimeError(f'Unrecognized layer {layer.__class__.__name__}')
        # Add named layers to our model
        model.add_module(name, layer)
        # If the name was included in content layer names, add content loss layer
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            # Add content loss layer to model
            model.add_module(f'content_loss_{i}', content_loss)
            # Add to content losses array
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            # Add style loss to our model
            model.add_module(f'style_loss_{i}', style_loss)
            # Add to style losses array
            style_losses.append(style_loss)
    # Iterate backwards through our model
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    # Only keep from the beginning of the model through the relevant layers
    model = model[: (i + 1)]
    return model, content_losses, style_losses


# Do the actual style transfer given the original (vgg) net and relevant images
def transfer_style(net, content_image, style_image, input_image, device):
    # Calculate the model we will use and losses
    model, content_losses, style_losses = get_style_model_and_losses(net, content_image, style_image, device)
    # We will be training the input image, not the model
    input_image.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_image])
    content_weight = 1
    style_weight = 1500000
    # Track progress in an array so inline function can increment
    counter = [0]
    n_steps = 300
    while counter[0] <= n_steps:
        def closure():
            # Clamp values of the image to stay between zero and one
            with torch.no_grad():
                input_image.clamp_(0, 1)
            optimizer.zero_grad()
            # Run the model on the image
            model(input_image)
            # Calculate the content and style
            content_score = 0
            style_score = 0
            for c in content_losses:
                content_score += c.loss
            for s in style_losses:
                style_score += s.loss
            # Losses are weighted and added
            content_score *= content_weight
            style_score *= style_weight
            loss = content_score + style_score
            loss.backward()
            counter[0] += 1
            return style_score + content_score
        optimizer.step(closure)
    with torch.no_grad():
        input_image.clamp_(0, 1)
    # Return the modified input image (which is now the output image)
    return input_image


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determine what size to make the image. Smaller is faster
    imsize = 128
    # Load content and style images
    print("Select Content Image")
    content_image = load_image(imsize, device)
    print("Select Style Image")
    style_image = load_image(imsize, device)
    # Content and style images should match for the style transfer to work
    assert style_image.size() == content_image.size()
    # Use pretrained vgg model to make our network
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    # Use the content image as a starting point
    input_image = content_image.clone()
    # Run the style transfer algorithm
    output_image = transfer_style(vgg, content_image, style_image, input_image, device)
    # Display the final result
    plt.ion()
    plt.figure()
    imshow(output_image, title='Output Image')
    plt.ioff()
    plt.show()
