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


# .xml을 .txt파일로 바꾸는 함수
##### yolo는 txt파일을 이용하기 때문에 xml파일은 사용할 수 없다.

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/b2bcfd6a-340e-4c2f-b84a-fe932e2eb8a0)

```
주어진 데이터는 coco데이터이다. yolo를 사용하려면 .xml파일을 .txt파일로 변경해야 한다.
```

```python
# .xml파일을 .txt파일로 바꾸는 함수를 이용한다.
def write_yolov8_txt(folder, annotation):
  out_filename = str(folder + annotation[0][:-3])
  out_filename = os.path.splitext(out_filename)[0]
  out_filename = out_filename+'.txt'
```

# coco데이터와 yolo데이터의 차이

![image](https://github.com/sesac-google-ai-1st/happy_friends/assets/147118232/500ffcde-95ef-4c83-88bf-111ff9972ecd)

```
"Coco"와 "YOLO"는 컴퓨터 비전 및 객체 감지와 관련된 두 가지 다른 개념입니다. 
"Coco"는 Microsoft이 개발한 Common Objects in Context(일반적인 맥락의 객체)의 약자로, 객체 감지 및 세분화를 위한 데이터셋을 나타냅니다. 
반면에 "YOLO"는 You Only Look Once의 약자로, 객체 감지를 수행하는 데 사용되는 딥러닝 알고리즘을 나타냅니다.

**Coco 데이터**

개요 : Coco 데이터셋은 객체 감지, 분할 및 캡션 생성을 위한 대규모 데이터셋으로, 이미지에 대한 다양한 객체의 주석이 포함되어 있습니다.
Coco 데이터 포맷은 bbox 값이 x, y, w, h 값으로 구성되어 있으며, 모든 영상의 주석이 담겨진 하나의 json 파일로 구성되어 있습니다.

내용 : Coco 데이터셋은 80가지 클래스에 대한 이미지와 각 이미지에서 발견된 객체에 대한 바운딩 박스 정보가 포함되어 있습니다.

활용 : 주로 객체 감지 및 세분화 모델의 학습과 평가에 사용됩니다.


**YOLO 데이터**

개요: YOLO는 객체 감지를 위한 신경망 알고리즘으로, 이미지를 한 번에 전체적으로 분석하여 객체를 감지하는 특징이 있습니다.
YOLO에서의 포맷은 클래스 번호와 전체 영상 크기에 대한 center x, center y, w, h 비율 값으로 구성되어 있으며, 한 영상 당 한 개의 txt파일로 구성되어 있습니다. 

내용: YOLO는 훈련을 위한 입력 이미지와 해당 이미지에서 발견된 객체의 바운딩 박스 및 클래스 라벨 정보를 필요로 합니다.

활용: YOLO는 실시간 객체 감지에 매우 효과적이며, 자율 주행 자동차, 보안 시스템 및 기타 응용 분야에서 활용됩니다.


요약하면, Coco는 객체 감지 및 세분화를 위한 데이터셋이고, YOLO는 이러한 데이터셋을 사용하여 훈련된 객체 감지 알고리즘을 나타냅니다.
Coco 데이터셋은 여러 응용 분야에서 사용될 수 있으며, YOLO는 실시간으로 객체를 감지하기 위한 효율적인 딥러닝 알고리즘입니다.
```

# coco데이터를 yolo데이터로 바꾸는 함수

```python
# coco데이터를 yolo데이터로 바꾸는 함수
def to_yolov8(y):
  """
  # change to yolo v8 format
  # [x_top_left, y_top_left, x_bottom_right, y_bottom_right] to
  # [x_center, y_center, width, height]
  """
  width = y[2] - y[0]
  height = y[3] - y[1]

  if width < 0 or height < 0:
      print("ERROR: negative width or height ", width, height, y)
      raise AssertionError("Negative width or height")
  return (y[0] + (width/2)), (y[1] + (height/2)), width, height

```

---

# 레이블 시각화 하기


## 레이블 시각화와 어노테이션의 개념

<span style="background-color:#fff5b1">레이블 시각화(label visualization)</span> : 데이터의 레이블 정보를 시각적으로 나타내어 데이터의 특징이나 패턴을 이해하거나 모델의 결과를 검증하는 데 도움을 주는 과정입니다. 
주로 컴퓨터 비전 분야에서 이미지나 비디오 데이터의 객체 감지, 분류, 세그멘테이션 등과 관련된 작업에서 사용됩니다.

**레이블 시각화의 방법**

<span style="background-color:#fff5b1">바운딩 박스 시각화 (Bounding Box Visualization)</span> : 객체 감지 작업에서 사용되는 바운딩 박스 정보를 시각적으로 나타냅니다. 이미지에 직사각형 상자를 그려 해당 객체의 위치를 표시합니다.

<span style="background-color:#fff5b1">클래스 레이블 시각화 (Class Label Visualization)</span> : 이미지나 객체에 대한 클래스 정보를 시각적으로 나타냅니다. 분류 작업에서 사용되며, 이미지 위에 클래스 레이블을 표시하는 방법이 있습니다.

<span style="background-color:#fff5b1">세그멘테이션 맵 시각화 (Segmentation Map Visualization)</span> : 이미지의 각 픽셀을 특정 클래스에 할당한 세그멘테이션 맵을 시각적으로 표현합니다. 객체의 윤곽이나 영역을 강조하여 보여줍니다.

<span style="background-color:#fff5b1">히트맵 시각화 (Heatmap Visualization)</span> : 특정 지역이나 객체에 대한 확률 또는 중요도를 나타내는 히트맵을 시각적으로 표현합니다. 특히 객체의 중요한 부분을 강조하기 위해 사용됩니다.

<span style="background-color:#fff5b1">포인트 레이블 시각화 (Point Label Visualization)</span> : 객체의 중심점이나 특정 포인트를 시각적으로 나타냅니다.

레이블 시각화는 데이터 이해, 모델 디버깅, 결과 해석, 모델 성능 평가 등 다양한 목적으로 사용되며, 시각적으로 명확하게 표현된 레이블은 모델 개발 및 향상에 도움을 줄 수 있습니다.



<span style="background-color:#fff5b1">어노테이션(annotation)</span> :  데이터에 대한 부가적인 정보를 제공하여 데이터를 이해하고 활용할 수 있도록 도와주는 작업을 말합니다. 
주로 이미지나 비디오 데이터에 대해 특정 객체나 특성을 나타내는 정보를 포함합니다. 
객체 감지나 세분화 등의 작업에서 어노테이션은 모델 학습에 사용되거나 모델의 결과를 검증하는 데 활용됩니다.
주로 컴퓨터 비전 분야에서 사용되며, 다양한 형태의 어노테이션이 있습니다.

**다양한 형태의 어노테이션**

<span style="background-color:#fff5b1">바운딩 박스 어노테이션 (Bounding Box Annotation)</span> : 객체를 감싸는 직사각형 상자를 정의합니다. 
주로 객체 감지와 같은 작업에서 사용됩니다.

<span style="background-color:#fff5b1">포인트 어노테이션 (Point Annotation)</span> : 객체의 중심점이나 특정 포인트를 나타내는 어노테이션입니다.

<span style="background-color:#fff5b1">세그멘테이션 어노테이션 (Segmentation Annotation)</span> : 이미지의 각 픽셀을 특정 클래스에 할당하여 객체의 윤곽을 정의합니다. 주로 이미지 세분화 작업에서 사용됩니다.

<span style="background-color:#fff5b1">클래스 레이블 어노테이션 (Class Label Annotation)</span> : 이미지나 객체에 대한 클래스 정보를 제공합니다. 분류 작업에서 사용됩니다.

어노테이션은 보통 수동으로 작업되며, 품질과 정확도는 데이터셋의 성능에 직접적인 영향을 미칩니다. 최근에는 더욱 정교한 어노테이션 작업을 위해 전문적인 툴과 서비스가 제공되고 있습니다.

"레이블 시각화"와 "어노테이션"은 밀접하게 관련되어 있지만, 몇 가지 차이가 있습니다.



어노테이션 (Annotation):

어노테이션은 데이터에 대한 부가 정보를 나타냅니다. 주로 객체 감지, 분류, 세그멘테이션 등과 관련된 작업에서 사용됩니다.
어노테이션은 주로 수동으로 작성되며, 데이터셋에 대한 실제 레이블 또는 태그를 의미합니다. 바운딩 박스 좌표, 클래스 레이블, 세그멘테이션 맵 등이 어노테이션에 속합니다.


레이블 시각화 (Label Visualization):

레이블 시각화는 어노테이션 정보를 시각적으로 나타내는 과정입니다. 어노테이션된 데이터의 특성을 이해하거나 모델의 결과를 검증하는 데 사용됩니다.
주로 이미지나 비디오 데이터에서 객체의 위치, 클래스, 윤곽 등을 시각적으로 나타냅니다. 바운딩 박스, 클래스 레이블, 세그멘테이션 맵 등의 시각화가 여기에 해당합니다.


요약하면, 어노테이션은 데이터에 대한 실제 정보를 나타내는 것이며, 레이블 시각화는 이 어노테이션 정보를 시각적으로 나타내어 데이터 이해와 모델 평가를 용이하게 합니다. 레이블 시각화는 어노테이션에 의존하여 데이터를 시각적으로 표현하는 단계라고 볼 수 있습니다.



