# Deep Learning Project

## How to Test Our Models Using media_utils.py

First, install all the libraries listed in `requirements.txt`. The command is:
```bash
pip3 install -r requirements.txt
```

Now, simply choose a model and run the command:
```bash
python3 media_utils.py model OPTIONS
```
## Examples
Creating the video featured in the presentation using the CNN:

```bash
python3 media_utils.py "models/landscape_128x128_30_color_consistency 92_08_.h5" -v --cc-loss -128 -V "test/videos/Video-test-cromavivo - original.mp4"
```
## Kaggle
The project is Kaggle-friendly. 

## License

This project is released under the MIT License.  
For further details, see the `LICENSE` file included in the repository.
