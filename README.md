# happy_friends
![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/107024182/0909b93f-39fb-4e3d-8318-5e71d0628117)
```
데이터 출처 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=164
```

```
학습목표 : 영동고속도로 ch01 - ch04 자료들을 yolo로 학습시켜 car,truck,bus를 구분하도록 만든다.
```

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/523939c7-0ba1-4341-864e-69f7151c113b)

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/c26726d7-ae58-49a3-84e8-a0936f7d9003)

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/88266b98-a53e-40e4-9f6d-1825534d3150)

```
영동 고속도로 일부를 조사
주어진 데이터는 coco데이터이다. yolo를 사용하려면 .xml파일을 .txt파일로 변경해야 한다.
```
![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/b2bcfd6a-340e-4c2f-b84a-fe932e2eb8a0)

```python
# .xml파일을 .txt파일로 바꾸는 함수를 이용한다.
def write_yolov8_txt(folder, annotation):
  out_filename = str(folder + annotation[0][:-3])
  out_filename = os.path.splitext(out_filename)[0]
  out_filename = out_filename+'.txt'
```

## coco데이터와 yolo데이터의 차이

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/500ffcde-95ef-4c83-88bf-111ff9972ecd)

```
"Coco"와 "YOLO"는 컴퓨터 비전 및 객체 감지와 관련된 두 가지 다른 개념입니다. 
"Coco"는 Microsoft이 개발한 Common Objects in Context(일반적인 맥락의 객체)의 약자로, 객체 감지 및 세분화를 위한 데이터셋을 나타냅니다. 
반면에 "YOLO"는 You Only Look Once의 약자로, 객체 감지를 수행하는 데 사용되는 딥러닝 알고리즘을 나타냅니다.

** Coco 데이터 **

개요 : Coco 데이터셋은 객체 감지, 분할 및 캡션 생성을 위한 대규모 데이터셋으로, 이미지에 대한 다양한 객체의 주석이 포함되어 있습니다. Coco 데이터 포맷은 bbox 값이 x, y, w, h 값으로 구성되어 있으며, 모든 영상의 주석이 담겨진 하나의 json 파일로 구성되어있습니다.

내용 : Coco 데이터셋은 80가지 클래스에 대한 이미지와 각 이미지에서 발견된 객체에 대한 바운딩 박스 정보가 포함되어 있습니다.

활용 : 주로 객체 감지 및 세분화 모델의 학습과 평가에 사용됩니다.


** YOLO 데이터 **

개요: YOLO는 객체 감지를 위한 신경망 알고리즘으로, 이미지를 한 번에 전체적으로 분석하여 객체를 감지하는 특징이 있습니다.YOLO에서의 포맷은 클래스 번호와 전체 영상 크기에 대한 center x, center y, w, h 비율 값으로 구성되어 있으며, 한 영상 당 한 개의 txt파일로 구성되어있습니다. 

내용: YOLO는 훈련을 위한 입력 이미지와 해당 이미지에서 발견된 객체의 바운딩 박스 및 클래스 라벨 정보를 필요로 합니다.

활용: YOLO는 실시간 객체 감지에 매우 효과적이며, 자율 주행 자동차, 보안 시스템 및 기타 응용 분야에서 활용됩니다.


요약하면, Coco는 객체 감지 및 세분화를 위한 데이터셋이고, YOLO는 이러한 데이터셋을 사용하여 훈련된 객체 감지 알고리즘을 나타냅니다. Coco 데이터셋은 여러 응용 분야에서 사용될 수 있으며, YOLO는 실시간으로 객체를 감지하기 위한 효율적인 딥러닝 알고리즘입니다.
```
