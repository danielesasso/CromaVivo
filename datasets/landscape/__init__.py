from os import listdir
from os.path import join, exists, isdir
from PIL import Image, ImageEnhance
import numpy as np

def color_preprocessing(image: Image, value: int=1) -> Image:
    """
    Preprocessa le immagini aumentando la saturazione del colore

    @param image: immagine da modificare
    @param value: valore che indica quanto viene modificata

    @return: immagine modificata
    """
    en = ImageEnhance.Color(image)
    return en.enhance(value)

def load_images(directory='.', limit=9999, img_shape=(128, 128)) -> None:
    """
    Carica le immagini dal dataset

    @param directory: directory su cui andare a prendere le immagini
    @param limit: limita le immagini da caricare
    @param img_shape: quando esegue il resize dell'immagine specifica la sua dimensione
    @return: dataset di immagini B/N e dataset di immagini RGB
    """
    x_ds = []
    y_ds = []
    files = listdir(directory)
    files.sort()
    for i, filename in enumerate(files[:limit]):
        image_path = join(directory, filename)
        try:
            rgb_img = Image.open(image_path).convert('RGB')
        except:
            print(f'skipped {image_path}')
            continue
        rgb_img = rgb_img.resize(img_shape)
        gray_img = rgb_img.convert('L')
        x_ds.append(np.array(gray_img))
        y_ds.append(np.array(rgb_img))
    x_ds = np.array(x_ds).astype('float32')/255.0
    x_ds = np.expand_dims(x_ds, axis=-1)
    y_ds = np.array(y_ds).astype('float32')/255.0
    return x_ds, y_ds

def get_dataset_dir():
    """
    Ricerca la directory dove è presente il dataset a seconda
    dell'ambiente in cui si trova: Colab, Kaggle, Locale.

    @return: stringa con il path della directory
    @raises NotADirectoryError: nessuna delle directory esiste
    """
    dirs = [
        '/content/drive/MyDrive/landscape',
        '/kaggle/input/landscape-pictures',
        'datasets/landscape'
    ]
    for d in dirs: # ricerca per ogni possibile directory
        if exists(d) and isdir(d): # quando trova la prima esistente
            return d # la restituisce
    else:
        raise NotADirectoryError('Dataset non trovato')

print('Caricamento immagini')
dataset_dir = get_dataset_dir()
print(f'Dataset trovato in {dataset_dir}')
gray_images, rgb_images = load_images(directory=dataset_dir)
print(f'Caricate {len(rgb_images)} immagini')
