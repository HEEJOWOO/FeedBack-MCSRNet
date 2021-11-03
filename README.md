# FeedBack Multi-Connection  Super Resolution Network



### Network Architecture

![0_2page](https://user-images.githubusercontent.com/61686244/129152667-385afc5f-17dd-439b-a972-95af90b3ce85.png)
![image](https://user-images.githubusercontent.com/61686244/140008138-02bbb0ab-a739-45a7-b2aa-f55b7421cd5e.png)

### Docker
<pre>
<code>
$ sudo docker pull heejowoo/feedback_mcsrnet:ver0.1
</code>
</pre>

<pre>
<code>
workspace
  ├──train_MCSRNet.py
  ├──test_MCSRNet.py
  ├──models.py
  ├──utils.py
  ├──FLOPs
  └──data
      ├── Set5
      ├── Set14
      ├── BSD100
      └── Urban100               
  └──BLAH_BLAH
      ├── DIV2K_x2.h5
      ├── DIV2K_x3.h5
      ├── DIV2K_x4.h5
      ├── Set5_x2.h5
      ├── Set5_x3.h5
      ├── Set5_x4.h5
      └── outputs
            └── x2
                 └── best.pth
            └── x3
                 └── best.pth  
            └── x4
                 └── best.pth            
</code>
</pre>


### Experiment Result
* PSNR for Test Set


|Dataset|Scale|SRCNN|RDN|RCAN|CARN|CBPN|OURS|
|-------|-----|-----|---|----|----|----|----|
|Set5(PSNR)|x2|36.66|38.24|38.27|37.76|37.90|38.22|
|Set5(PSNR)|x3|32.75|34.71|34.74|34.29|-|xxx|
|Set5(PSNR)|x4|30.48|32.47|32.63|32.13|32.21|32.6|
|Set14(PSNR)|x2|32.45|34.01|34.12|33.52|33.60|34.03|
|Set14(PSNR)|x3|29.29|30.57|30.65|30.29|-|xxx|
|Set14(PSNR)|x4|27.50|28.81|28.87|28.60|28.63|28.93|
|BSD100(PSNR)|x2|31.36|32.34|32.41|32.09|32.17|32.37|
|BSD100(PSNR)|x3|28.41|29.26|29.32|29.06|-|xxx|
|BSD100(PSNR)|x4|26.90|27.72|27.77|27.58|27.58|27.8|
|Urban100(PSNR)|x2|29.51|32.89|33.34|31.92|32.14|32.86|
|Urban100(PSNR)|x3|26.24|28.80|29.09|28.06|-|xxx|
|Urban100(PSNR)|x4|24.52|26.61|26.82|26.07|26.14|26.82|



 ### Params & Multi-Adds

|x4|SRCNN|RDN|RCAN|CARN|CBPN|OURS|
|--|-----|---|----|----|----|----|
|Parameter(M)|0.057|22|16|1.5|1.1|2.28|
|Multi-Adds(G)|52.7|1,309|919.9|90.9|97.9|1,188|


