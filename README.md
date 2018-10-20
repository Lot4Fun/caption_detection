# __Telop Detection__

## Description
IN PRODUCTION: Detect telop in image.

## Demo
```
python impulso.py predict -e XXXX-XXXX-XXXX -m X -x X=DIR -y Y-DIR
```

## Results
Not successfully detected.  
- Heatmap: Score of each pixel
- Rectangle: Detected bounding box

![Sample01](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample01.jpg)
![Sample02](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample02.jpg)
![Sample03](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample03.jpg)
![Sample04](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample04.jpg)
![Sample05](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample05.jpg)
![Sample06](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample06.jpg)
![Sample07](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample07.jpg)
![Sample08](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample08.jpg)
![Sample09](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample09.jpg)
![Sample10](https://github.com/pystokes/CTPN/blob/master/tmp/output/figures/sample10.jpg)

## Requirement
tensorflow-gpu==1.4.0  
Keras==2.1.4  

## Install
```
git clone https://github.com/pystokes/CTPN.git
```

## Usage
### Create dataset
```
python impulso.py dataset
```

### Prepare
```
python impulso.py prepare -d DATA-ID
```

### Train
To resume training, specify MODEL-ID.
```
python impulso.py train -e EXPERIMENT-ID [-m MODEL-ID]
```

### Test
```
python impulso.py test -e EXPERIMENT-ID -m MODEL-ID
```

### Predict
```
python impulso.py predict -e EXPERIMENT-ID -m MODEL-ID -x INPUT_DIR -y OUTPUT_DIR
```

## License
- Permitted: Private Use  
- Forbidden: Commercial Use  

## Author
[LotFun](https://github.com/pystokes)
