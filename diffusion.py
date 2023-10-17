import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import show_tensor_image

class Diffusion():
    def __init__(self, img_height, img_width, device, beta_start=1e-4, beta_end=0.02, beta_num=1000):
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, beta_num).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)


    def add_noise(self, images, t):
        alphas_hat_t = self.alphas_hat.gather(-1, t.to(self.device)).reshape(-1, 1, 1, 1)
        eps = torch.randn_like(images)
        mean = alphas_hat_t ** 0.5 * images
        variance = (1. - alphas_hat_t) ** 0.5 * eps
        return mean + variance, eps


    @torch.no_grad()
    def sample_timestep(self, model, noised_image, t):
        model.eval()
        #print(t)
        predicted_noise = model(noised_image, t)
        model.train()
        if t == 0:
            noise = torch.zeros_like(noised_image)
        else:
            noise = torch.randn_like(noised_image)
        return 1 / (self.alphas[t])**0.5 * (noised_image - ((1 - self.alphas[t]) / (1 - self.alphas_hat[t]) ** 0.5) * predicted_noise) + self.betas[t]**0.5 * noise


    def denoise_one_step(self, noised_images, noise, t):
        alphas_hat_t = self.alphas_hat.gather(-1, t.to(self.device)).reshape(-1, 1, 1, 1)
        mean = alphas_hat_t ** 0.5
        variance = (1. - alphas_hat_t) ** 0.5
        return (noised_images - variance * noise) / mean 


    @torch.no_grad()
    def sample_random_one_step(self, model, T, train_dataset):
        idx = torch.randint(len(train_dataset), (1,))
        random_image = train_dataset[idx][0]
        random_image = random_image.unsqueeze(0).to(self.device)
        t = torch.tensor([T]).long()
        noised_image, noise = self.add_noise(random_image, t)
        plt.figure(figsize=(15,15))
        plt.axis('off')

        plt.subplot(1, 3, 1)
        show_tensor_image(random_image.detach().cpu())
        plt.title('original')

        plt.subplot(1,3,2)
        show_tensor_image(torch.clamp(noised_image, -1, 1).detach().cpu())
        plt.title(f'Noise t = {T}')

        predicted_noise = model(noised_image, t)
        image = self.denoise_one_step(noised_image, predicted_noise, t)
        #image = torch.clamp(image, -1, 1)
        plt.subplot(1,3,3)
        show_tensor_image(torch.clamp(image, -1, 1).detach().cpu())
        plt.title('predicted')
        plt.show()


    @torch.no_grad()
    def sample_image(self, model, T):
        image = torch.randn((1,3,self.img_height, self.img_width), device=self.device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T/num_images)

        plt.subplot(1, num_images + 1, num_images + 1)
        show_tensor_image(image.detach().cpu())
        plt.title(num_images + 1)

        for i in range(0,T)[::-1]:
            t = torch.tensor([i]).long().to(self.device)
            image = self.sample_timestep(model, image, t)
            # Edit: This is to maintain the natural range of the distribution
            #image = torch.clamp(image, -1.0, 1.0)
            if i == 0:
                plt.subplot(1, num_images + 1, int(i/stepsize)+1)
                image = torch.clamp(image, -1.0, 1.0)
                #show_tensor_image(image.detach().cpu())
                show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                plt.title(int(i/stepsize)+1)
                break
            if i % stepsize == 0:
                plt.subplot(1, num_images + 1, int(i/stepsize)+1)
                #show_tensor_image(image.detach().cpu())
                show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                plt.title(int(i/stepsize)+1)
        plt.show()   

    @torch.no_grad()
    def sample_random(self, model, T, train_dataset):
        idx = torch.randint(len(train_dataset), (1,))
        random_image = train_dataset[idx][0]
        random_image = random_image.unsqueeze(0).to(self.device)
        noised_image, noise = self.add_noise(random_image, torch.tensor([T]).long())
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T/num_images)
        

        plt.subplot(1, num_images + 1, int(T/stepsize) + 1)
        show_tensor_image(random_image.detach().cpu())
        plt.title(int(T/stepsize) + 1)

        image = noised_image

        for i in range(0,T)[::-1]:
            t = torch.tensor([i]).long().to(self.device)
            image = self.sample_timestep(model, image, t)
            # Edit: This is to maintain the natural range of the distribution
            #image = torch.clamp(image, -1.0, 1.0)
            if i == 0:
                plt.subplot(1, num_images + 1, int(i/stepsize)+1)
                image = torch.clamp(image, -1.0, 1.0)
                #show_tensor_image(image.detach().cpu())
                show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                plt.title(int(i/stepsize)+1)
                break
            if i % stepsize == 0:
                plt.subplot(1, num_images + 1, int(i/stepsize)+1)
                #show_tensor_image(image.detach().cpu())
                show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                plt.title(int(i/stepsize)+1)
        plt.show()
        

    @torch.no_grad()
    def sample_random_multiple(self, model, T, train_dataset, num=5):
        for val in range(num):
            idx = torch.randint(len(train_dataset), (1,))
            random_image = train_dataset[idx][0]
            random_image = random_image.unsqueeze(0).to(self.device)

            #show_tensor_image(random_image.detach().cpu())
            #plt.show()
            #exit()

            noised_image, noise = self.add_noise(random_image, torch.tensor([T]).long())
            #plt.figure(figsize=(2,2))
            plt.axis('off')
            num_images = 10
            stepsize = int(T/num_images)
            

            plt.subplot(num, num_images + 1, val * (num_images+1) + int(T/stepsize) + 1)
            show_tensor_image(random_image.detach().cpu())
            plt.title(str(val * (num_images + 1) + int(T/stepsize) + 1))

            image = noised_image

            for i in range(0,T)[::-1]:
                t = torch.tensor([i]).long().to(self.device)
                image = self.sample_timestep(model, image, t)
                # Edit: This is to maintain the natural range of the distribution
                #image = torch.clamp(image, -1.0, 1.0)
                if i == 0:
                    plt.subplot(num, num_images + 1, val * (num_images+1) + int(i/stepsize)+1)
                    image = torch.clamp(image, -1.0, 1.0)
                    show_tensor_image(image.detach().cpu())
                    plt.title(str(val * (num_images+1) + int(i/stepsize)+1))
                    break
                if i % stepsize == 0:
                    plt.subplot(num, num_images + 1, val * (num_images+1) + int(i/stepsize)+1)
                    #show_tensor_image(image.detach().cpu())
                    show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                    plt.title(str(val * (num_images+1) + int(i/stepsize)+1))
        
        plt.show()

    @torch.no_grad()
    def sample_image_multiple(self, model, T, num = 5):
        for val in range(num):
            image = torch.randn((1,3,self.img_height, self.img_width), device=self.device)
            
            

            plt.axis('off')
            num_images = 10
            stepsize = int(T/num_images)

            plt.subplot(num, num_images + 1, val * (num_images+1) + int(T/stepsize) + 1)
            show_tensor_image(image.detach().cpu())
            plt.title(str(val * (num_images + 1) + int(T/stepsize) + 1))

            

            for i in range(0,T)[::-1]:
                t = torch.tensor([i]).long().to(self.device)
                image = self.sample_timestep(model, image, t)
                # Edit: This is to maintain the natural range of the distribution
                #image = torch.clamp(image, -1.0, 1.0)
                if i == 0:
                    plt.subplot(num, num_images + 1, val * (num_images + 1) + int(i/stepsize)+1)
                    image = torch.clamp(image, -1.0, 1.0)
                    show_tensor_image(image.detach().cpu())
                    plt.title(val * (num_images + 1) + int(i/stepsize)+1)
                    break
                if i % stepsize == 0:
                    plt.subplot(num, num_images + 1, val * (num_images + 1) + int(i/stepsize)+1)
                    #show_tensor_image(image.detach().cpu())
                    show_tensor_image(torch.clamp(image, -1.0, 1.0).detach().cpu())
                    plt.title(val * (num_images + 1) + int(i/stepsize)+1)

        plt.show()

    @torch.no_grad()
    def generate_images(self, model, T, rows=5, cols= 5):
        for val in range(rows * cols):
            image = torch.randn((1,3,self.img_height, self.img_width), device=self.device)
            for i in range(0,T)[::-1]:
                t = torch.tensor([i]).long().to(self.device)
                image = self.sample_timestep(model, image, t)
                # Edit: This is to maintain the natural range of the distribution
                #image = torch.clamp(image, -1.0, 1.0)
                if i == 0:
                    plt.subplot(rows, cols, val+1)
                    image = torch.clamp(image, -1.0, 1.0)
                    show_tensor_image(image.detach().cpu())
                    plt.title(val + 1)
                    plt.axis('off')
                    break
        plt.show()

    @torch.no_grad()
    def generate_images_one_step(self, model, T, rows=5, cols= 5):
        t = torch.tensor([T]).long()
        for val in range(rows * cols):
            image = torch.randn((1,3,self.img_height, self.img_width), device=self.device)
            predicted_noise = model(image, t)
            image = self.denoise_one_step(image, predicted_noise, t)
            #image = torch.clamp(image, -1, 1)
            plt.subplot(rows, cols , val + 1)
            show_tensor_image(torch.clamp(image, -1, 1).detach().cpu())
            plt.title(f'gen {val + 1}')

        plt.show()

    @torch.no_grad()
    def compare_generate_images(self, model, T, rows=5, cols= 5):
        for val in range(rows * cols):
            gen_image = torch.randn((1,3,self.img_height, self.img_width), device=self.device)
            t = torch.tensor([T]).long().to(self.device)

            predicted_noise = model(gen_image, t)
            image = self.denoise_one_step(gen_image, predicted_noise, t)
            #image = torch.clamp(image, -1, 1)
            plt.subplot(rows, cols * 2 , 2*val + 1)
            show_tensor_image(torch.clamp(image, -1, 1).detach().cpu())
            plt.title(f'1step {val + 1}')

            image = gen_image
            for i in range(0,T)[::-1]:
                t = torch.tensor([i]).long().to(self.device)
                image = self.sample_timestep(model, image, t)
                # Edit: This is to maintain the natural range of the distribution
                #image = torch.clamp(image, -1.0, 1.0)
                if i == 0:
                    plt.subplot(rows, cols * 2, 2*val + 2)
                    image = torch.clamp(image, -1.0, 1.0)
                    show_tensor_image(image.detach().cpu())
                    plt.title(f'multi step {val + 1}')
                    break
        plt.show()