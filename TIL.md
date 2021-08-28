# TIL

그날 그날 Check 하고 싶은 부분 간략 정리

>보통  
Deep Learning : 딥러닝 텐서플로 교과서 베이스로 공부, [관련 repo](https://github.com/leemik3/tensorflow-2.0)  
Algorithm : 백준 문제집, [관련 repo](https://github.com/hymk-scdc/algorithm/tree/main/mk)  
Python : 딥러닝, 알고리즘 공부하면서 몰랐던 파이썬 문법 정리, [관련 repo](https://github.com/leemik3/python)

## 2021.08.28
#### [Algorithm]
1. .isupper(), .islower(), .isdigit(), .isspace()
2. ```if 'a' <= c <= 'z':``` 사용 가능
3. 탑다운은 계속 시간초과가 남. 보텀업이 보통인듯?

## 2021.08.27
#### [Deep Learning]
1. [Fine-grained 개념](https://github.com/leemik3/tensorflow-2.0/wiki/Coarse-grained,-Fine-grained)
2. [Grid Search 개념](https://github.com/leemik3/tensorflow-2.0/wiki/Grid-Search)
3. [FLOPS](https://github.com/leemik3/tensorflow-2.0/wiki/FLOPS) : 컴퓨터의 성능을 수치로 나타낸 단위

## 2021.08.26
#### [Deep Learning]
1. Variational AutoEncoder의 늪에서 허우적허우적~ 그래 쉬운 것만 공부하면 실력이 어떻게 늘겠어~ 어려운 거 꾸역꾸역 이해해나가야 몰랐던 걸 알게 되는거고 그게 반복돼야 실력이 느는거지~ 

#### [Algorithm]
2. 동적 계획법 (Dynamic Programming)

## 2021.08.25
#### [Algorithm]
1. shallow 복사 : 슬라이싱으로


## 2021.08.23
#### [Deep Learning]
1. 오늘 미팅 Keyword : Context Aware, Anomaly Detection, Autoencoder, stream data, CNN / RNN, self-supervised learning
2. Autoencoder을 anomaly 데이터 처리하는 데에 사용하기도 함. 차원을 압축시키는 과정에서 아무래도 noise 한 부분이 사라질 수 있기 때문.

#### [Python]
3. 
```result = list([0, 0, 0, 0] for i in range(len(N)))```
```result = list(zero for i in range(len(N)))```
: result[0][0]+=1 를 했을 때 위 코드에서는 정상 작동, 아래 코드에서는 모든 원소의 0번 인덱스 값이 바뀜

## 2021.08.22
#### [Algorithm]
1. [백준 시간 초과 원인](https://www.acmicpc.net/problem/15552)  
- Python  
  - 원인 : for문 문제를 풀 때 입출력 방식이 느리면 여러 줄을 입력/출력할 때 시간 초과가 날 수 있다
  - 해결 : input 대신 sys.stdin.readline을 사용.  단, 맨 끝의 개행문자까지 입력받으므로 문자열을 저장할 땐 .rstrip()을 추가로 해 주는 것이 좋다.
- 결론 : readline이나 PyPy로 제출하면 대부분 해결
2. [백준 문제 풀이 시 유의사항](https://www.acmicpc.net/blog/view/55)
3. 10828번 문항 
```
import sys
input = sys.stdin.readline
```
위에 추가해서 해결
4. ```if 문자열 == 문자열``` , ```if 문자열[:숫자] == 문자열``` : 연산 시간 차이가 많이 나나?  
5. list에서 인덱싱 슬라이싱보다 append, pop이 빠름
6. ```for i in range( int(input()) )``` 형식도 가능함 - 숏코딩
7. ```for i in range(4): print(~i)``` : -1 -2 -3 .. 으로 출력됨
8. ```zfill(width)``` ```rjust(width,[fillchar]```

## 2021.08.19
#### [Deep Learning]
1. ```.shuffle(buffer_size=xxx)``` : buffer_size 만큼 가져와서 shuffle 
2. ```.batch(batch_size)``` : batch_size 크기로 묶는다.

#### [Algorithm]
1. 내장 함수를 어느 정도까지 써도 되는지 모르겠다
2. 시간 초과 - 어떤 로직이 시간복잡도가 높은지 파악하는 것은... 계속 숙제

## 2021.08.18
#### [Deep Learning]
1. 오토인코더와 변형오토인코더 논문 읽고 이해하기
2. ```def __init__(self, **kwargs):``` : 딕셔너리 형태로 입력받기 위함 [wiki 참조](https://github.com/leemik3/python/wiki/*args---**kwargs)
3. ```@tf.function```

## 2021.08.17
#### [Algorithm]
1. map 함수
2. try, except, finally 정리

## 2021.08.13
#### [Deep Learning]
1. Inception, GoogLeNet 다시 정리함 : Sparse connection 효과를 주는 inception module (1x1, 3x3, 5x5이 각각 크고 작은 region을 커버)
2. 1과 관련해서 filter concatenation의 의미와 정확한 효과
3. globally connected layer과 locally connected layer의 차이 : 

## 2021.08.12
#### [Deep Learning]
1. ```tf.keras.layers.Input(shape=(784,))``` : 입력의 크기 (784, ) 어떤게 784라는 소리인지? 
2. ```np.prod(array)``` : array 를 product

## 2021.08.10
#### [Deep Learning]
1. ```glob.glob('<경로>'')``` : 해당 경로에 있는 모든 폴더 및 파일을 **리스트**로 반환

## 2021.08.09
#### [Deep Learning]
1. ```tf.keras.callbacks.ModelCheckpoint``` 콜백 함수 : 훈련 중간 / 마지막에 체크포인트 사용
2. ```sequence.pad_sequences(x_train, maxlen=maxlen)``` 0으로 시퀀스를 채움

## 2021.08.07
#### [Deep Learning]
1. ```tfds.load``` : ProfetchDataset,, for 문으로 출력하면 내용 볼 수 있음
2. ```padded_batch()``` : 배치에서 가장 긴 문자열의 길이를 기준으로 시퀀스를 0으로 채움
3. ```shuffle``` : ??

## 2021.08.06 
#### [Deep Learning]
1. ```Dense(64, input_shape=(4,), activation='relu')```  
- 유닛이 64개인건 이해 됨.
- (입력층이) (4,0) 형태를 가진다고 책에 써있음. 처음에는 4가 batch size, 데이터 크기인가 싶었는데, 데이터 1개가 (4,0) 형태,, 컬럼 개수인가? 싶었음... 계속 헷갈림 이게

## 2021.08.05
#### [Python]
1. 가정설정문 assert : 뒤의 조건이 거짓일 경우 에러 발생시킴
2.  __call__ : [매직 메소드](https://github.com/leemik3/python/wiki/%ED%81%B4%EB%9E%98%EC%8A%A4(class))

#### [Deep Learning]
3. ```tf.keras.preprocessing.sequence.pad_sequences```  
   : Transforms a **list** (길이 : num_samples) of sequences (lists of integers)   
   into a **2D Numpy array** of shape (num_samples, num_timesteps)
4. ```tf.unstack()```
5. Tensorslicedataset, Batchdataset, onehotencoding 등 적재적소 데이터 전처리에 대한 이해
6. '07_3_RNN_Cell.py' 클래스 부분 이해 안 감
7. ```VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])```  
: 경고 무시했음, 동작은 됨
   
## 2021.08.01
#### [Deep Learning]
1. 공간 피라미드 풀링, R-CNN, Fast R-CNN, Faster R-CNN 잘 이해되지 않았음... 논문을 읽어봐?

2. 완전연결층의 한계 : 고정된 크기의 입력만 받아들이며, 완전연결층을 거친 후에는 위치 정보가 사라진다. 

## 2021.07.31
#### [Deep Learning]
1. OOM (Out Of Memory)   
: 학습시키지 않고.. 근데 compile 생략했는데 compile과 train 차이,,?    
   가중치 로드하면 안 뜰 줄 알았는데 여전히 OOM 뜸

2. 1x1 합성곱의 의미? 무의미하다고 생각했음   
: 이미지 크기는 변함이 없는 게 맞음, 채널 수를 줄여서 파라미터를 줄여주는 효과

3. ResNet의 Residual : shortcut, 기울기 소실 문제를 해결하는 방식의 수학적인 부분이 잘 이해되지 않음

4. MaxPooling layer 를 거치면 spatial information 정보가 손실됨

## 2021.07.30
#### [Deep Learning]
1. ```MaxPooling2D(data_format='channels_last)``` : 입력 형식을 설정하는 파라미터   
channels_last : 입력 데이터 형식이  (배치 크기, 높이, 너비, 채널개수)  
channels_first : 입력 데이터 형식이 (배치 크기, 채널개수, 높이, 너비)
   
2. strides=2 와 strides=(2,2)
: Conv2D에서 차이 없는 듯? 정확히

3. ```flow_from_directory```   
: return (x,y)   
x : (batch_size, target_size, channels) 크기의 이미지  
y : labels

## 2021.07.29
#### [Python]
1. ```super().__init__()``` 
[클래스, 상속, 오버라이딩의 개념](https://github.com/leemik3/python/wiki/%ED%81%B4%EB%9E%98%EC%8A%A4(class))

2. 경로 - 마지막에 '/' 여부, 상대 경로와 절대 경로가 어떤 차이?
- log_dir = '../../data/chap6/img/log6-2/'   
- log_dir = '../../data/chap6/img/log6-2'   
- log_dir = 'D:\git\tensorflow-2.0\data\chap6\img\log6-1\train   

#### [Deep Learning]
3. ```Conv2D(kernel_initializer='he_normal')```
[가중치 초기화 방법 개념](https://github.com/leemik3/tensorflow-2.0/#%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94-%EB%B0%A9%EB%B2%95)
   
4. ```Conv2D(padding='valid)```   
padding='valid' : 패딩 없음   
padding='same' : 입력과 출력의 크기가 같도록 패딩



