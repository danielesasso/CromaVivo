import cv2
from sys import exit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os import mkdir
from os.path import join, exists, basename
from PIL import Image
from traceback import print_exc

model = None
img_size = None

def explode_video(video_path, convert_to_gray=True):
    global img_size
    """
    Separates all video frames for each picture

    @param video_path: video path
    @param convert_to_gray: indicates to convert all frames to grey

    @return: lista di frame
    """
    def preprocess_frame(frame, convert_to_gray=True):
        """
        Modifica il frame eseguendo un resize, trasformandolo in grigio 
        e trasformando in array float32 e normalizzandolo fra 0 e 1

        @param frame: immagine da modificare
        @param convert_to_gray: deve convertirlo in grigio?

        @return: frame modificato
        """
        frame = cv2.resize(frame, img_size)
        if convert_to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.array(frame).astype('float32')/255.0
    
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f'{video_path} non aperto')
        exit(1)

    success_code, frame = video.read() #leggo il primo frame
    frame_list = []

    while success_code:
        frame_list.append(preprocess_frame(frame, convert_to_gray=convert_to_gray))
        success_code, frame = video.read()

    video.release()

    return np.array(frame_list)

def create_output(frame_list, output_path, fps=30):
    global img_size
    """
    Crea il video risultante dalle predizioni del modello

    @param frame_list: lista di frame
    @param output_path: dove salvare il video
    @param fps: frame per secondo del video

    @return: None
    """
    (w, h) = img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    for frame in frame_list:
        tmp = (frame * 255).astype('uint8')
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        video_writer.write(tmp)

    video_writer.release()

def print_imgs(*arr_imgs: list, graph_shape: tuple[int, int]=(20, 10)) -> None:
    """
    Stampa più immagini per la comparazione.
    La prima immagine sarà sempre in bianco e nero. L'unico modo
    per non stampare la prima immagine in B/N è usare None come
    primo parametro.

    Parametri:
    *arr_imgs (list): le varie numpy array
    graph_shape (tuple[int, int]): forma del grafico risultante
    """
    if len(arr_imgs) == 0:
      print('Se volevi stampare il nulla, eccolo qua:')
      return
    
    l = len(arr_imgs) # numero liste fornite
    n = len(arr_imgs[-1]) # numero immagini per lista fornite

    fig, axs = plt.subplots(nrows=l, ncols=n, figsize=graph_shape) # grafico
    for j in range(l):
      for i in range(n):
        if j == 0 and arr_imgs[j] is not None: 
          axs[j,i].imshow(arr_imgs[j][i], cmap='gray')
        elif arr_imgs[j] is not None:
          axs[j,i].imshow(arr_imgs[j][i])
        axs[j,i].axis('off')
    plt.show()

def SSIM(y_true, y_pred):
    """
    Funzione di similarità del modello in quanto la richiede per il corretto caricamento
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def get_arguments():
    """
    Ottiene gli argomenti

    @return: lista degli argomenti 
    """
    a = ArgumentParser()
    a.add_argument('model', nargs=1, help='Il modello da utilizzare')
    a.add_argument('-i', '--images', nargs='+', dest='imgs', help='Le immagini da colorare')
    a.add_argument('-V', '--videos', nargs='+', dest='videos', help='I video da colorare')
    a.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='Attiva l\'output')
    a.add_argument('--ssim', action='store_true', dest='ssim', help='Usa la funzione SSIM')
    a.add_argument('--cc-loss', action='store_true', dest='cc_loss', help='Usa la funzione color_consistency_loss')
    a.add_argument('-64', action='store_true', dest='small', help='Imposta la dimensione delle immagini a 64x64')
    a.add_argument('-128', action='store_true', dest='medium', help='Imposta la dimensione delle immagini a 128x128')
    a.add_argument('-256', action='store_true', dest='large', help='Imposta la dimensione delle immagini a 256x256')

    return a.parse_args()

def make_video(input_path, verbose=True):
    global model
    assert model is not None
    try:
        mkdir(join('output', 'videos'))
    except:
        pass
    print(f'Carico')
    frames = explode_video(input_path)
    frames = np.expand_dims(frames, axis=-1)
    try:
        new_frames = model.predict(frames, verbose=1 if verbose else 0)
    except:
        print('Risorse esaurite')
        return
    create_output(new_frames, join('output', 'videos', basename(input_path)))

def make_images(input_paths, verbose=True):
    global model
    global img_size
    assert model is not None
    try:
        mkdir(join('output', 'images'))
    except:
        pass
    print(f'Carico')
    images = [np.array(Image.open(img).convert('L').resize(img_size)).astype('float32')/255.0 for img in input_paths]
    images = np.expand_dims(np.array(images), axis=-1)
    try:
        new_images = model.predict(images, verbose=1 if verbose else 0)
    except:
        print('Risorse esaurite')
        return
    for i, img in enumerate(new_images):
        s_img = (img * 255.0).astype('uint8')
        s_img = Image.fromarray(s_img)
        s_img.save(join('output', 'images', basename(input_paths[i])))

def color_consistency_loss(y_true, y_pred):
    # Calcola l'errore medio quadratico (MSE)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Calcola la deviazione standard dei canali di colore
    std_true = tf.math.reduce_std(y_true, axis=-1)
    std_pred = tf.math.reduce_std(y_pred, axis=-1)

    # Applica pesi ai termini di loss
    mse_weight = 0.92
    std_weight = 0.08

    # Calcola la funzione di loss combinata
    loss = mse_weight * mse + std_weight * tf.reduce_mean(tf.square(std_true - std_pred))

    return loss

if __name__ == '__main__':
    args = get_arguments()

    try:
        mkdir('output')
    except:
        pass

    assert exists(args.model[0])
    print(f'Carico modello {args.model[0]}')
    if args.ssim:
        model = tf.keras.models.load_model(args.model[0], custom_objects={'SSIM':SSIM})
    elif args.cc_loss:
        model = tf.keras.models.load_model(args.model[0], custom_objects={'color_consistency_loss':color_consistency_loss})
    else:
        model = tf.keras.models.load_model(args.model[0])
    print('Caricato')
    if args.small:
        print('Uso dimensione 64x64')
        img_size = (64, 64)
    elif args.medium:
        print('Uso dimensione 128x128')
        img_size = (128, 128)
    elif args.large:
        print('Uso dimensione 256x256')
        img_size = (256, 256)
    else:
        print('Non è stata specificata la dimensione')
    if args.imgs is not None:
        make_images(args.imgs, verbose=args.verbose)
    if args.videos is not None:
        for vid in args.videos:
            make_video(vid, verbose=args.verbose)
