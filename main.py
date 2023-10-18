import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import UNet
from diffusion import Diffusion
from utils import get_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 32
IMG_HEIGHT = 208
IMG_WIDTH = 176

model_path = 'D:\python fun projects\machine learning\models\celebdiffusion218x178_48rf(good_old).pth'
data_path = 'D:\python fun projects\machine learning\data'

num_epochs = 1000000
batch_size = 1
learning_rate = 1e-5
counter = 1000

def train():
    train_dataset, train_loader = get_data(IMG_HEIGHT, IMG_WIDTH, batch_size, data_path)

    model = UNet(EMBED_DIM, DEVICE).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path))
        print('Model Loaded')
    except:
        print('No model found -> Creating new model')
        pass

    diffusion = Diffusion(IMG_HEIGHT, IMG_WIDTH, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    l1 = nn.L1Loss()
    n_total_steps = len(train_loader)
    T = 999
    train_size = len(train_dataset)

    #Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for step, images in enumerate(train_loader):
            #print(images.shape)
            images = images[0].cuda()
            #labels = labels.cuda()

            t = torch.randint(0, T, (batch_size if step < n_total_steps -1 else train_size - batch_size * step,), device=DEVICE).long()
            noised_images, noise = diffusion.add_noise(images, t)

            with torch.cuda.amp.autocast():
                predicted_noise = model(noised_images, t)
                loss = l1(noise, predicted_noise)

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            print(f"\rEpoch {epoch:04d} | step {step:08d} Loss: {loss.item():<10.4f} with t={t[0]:<10d}", end='')

            if step % counter == 0:    
                torch.save(model.state_dict(), model_path)
                print('')

def test():
    '''
    Uncomment these to test your model
    '''
    train_dataset, train_loader = get_data(IMG_HEIGHT, IMG_WIDTH, batch_size, data_path)

    model = UNet(EMBED_DIM, DEVICE).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path))
        print('Model Loaded')
    except:
        print('No model found -> Creating new model')
        pass

    diffusion = Diffusion(IMG_HEIGHT, IMG_WIDTH, DEVICE)

    #diffusion.sample_random(model, T = 300, train_dataset=train_dataset)
    diffusion.sample_image(model, T=900)
    #diffusion.sample_image_multiple(model, T=900, num=10)
    #diffusion.sample_random_multiple(model, T=300, train_dataset=train_dataset)
    #diffusion.generate_images(model, T=900, rows=2, cols=2)
    #diffusion.sample_random_one_step(model, T=300, train_dataset=train_dataset)
    #diffusion.generate_images_one_step(model, T=900, rows=3, cols=3)
    #diffusion.compare_generate_images(model, T=900, rows=2, cols=2)
    

if __name__ == '__main__':
    train()
    #test()
    