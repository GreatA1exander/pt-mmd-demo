# PT-MMD: A Novel Statistical Framework for the Evaluation of Generative Systems

This repository is the code demo of PT-MMD, as shown in our paper located at https://arxiv.org/abs/1910.12454

Instructions:

 - Install `tensorflow-gpu==1.14` `torch` `torchvision` `pywavelets` `matplotlib` `numpy` `opencv-python` `pillow` `tqdm`
 - Make sure that you have an up-to-date NVIDIA GPU and the appropriate drivers.
 - You will need several files if you wish to run the generation code. They are available at https://drive.google.com/drive/folders/1td3etOgmeND-t_2ibzlETrqVUUD5v9Eh
 - Please place the .pth files in the ./GAN/WGAN/ folder.
 - Please place the .pkl files in the ./GAN/PGAN/ folder.
 - From there, run the appropriate driver notebook in each of those folders to generate synthetic data that will be stored in a .bin file. *NOTE: You can skip this and use our supplied .bin files in the same folder.*
 - Move the two synthetic .bin files into the ./common/data/ folder.
 - Download the ground truth data from the same link (lsun_rgb_samples.bin) and place it into the data folder as well.
 
At this point you can run the PT-MMD Driver Demo python notebook. After running that notebook, you should receive similar results to our results for the GANS at the end of our paper. Feel free to use this framework for your own experiments! :)
