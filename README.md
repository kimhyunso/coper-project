# coper-project

# 목표
실시간 채팅을 분석하여 반응을 확인하는 것

# 회의록 

## 230210 금 
- 첫 팀 빌딩 
- 팀 명 : 도원결의 
- 현재 확정 멤버  ```이도원``` ```김현수 ``` ```김주환```
- 주제 : YOUTUBE API와 댓글을 이용한 유튜브 영상 및 채널 분석 

## 230211 토 
- 오후 2시 ~ 오후 5시 (약 3시간 오버 )

> 프로젝트의 큰 흐름 논의
YOUTUBE API > 댓글 정보 수집 > konlypy 형태소분석기, TD-IDF를 활용해 단어에 가중치 부여, 핵심키워드 파악 > 혼동행렬, 데이터 쏠림현상 고려해서 logistic regression 머신러닝 학습 > 각종 수치 시각화 

> 초기 개발 환경 셋팅 ( 필요 패키지 다같이 설치 )
```
- 설치 순서
1. JDK 17.0.1 설치 [설치 시 PATH설정 체크 필히 누를 것.]
2. JPype 
3. JPype의 PATH를 등록하고 컴퓨터 재시작
4. sentimental_anlaysis 파일에서 요구하는 패키지 전부 설치.
예시) pip install pandas
 - pandas
 - numpy
 - matplotlib
 - seaborn
 - sci-kit learn 
파일 실행해서 동작 확인.
```
***프로젝트 진행하는데 핵심이 되는 패키지***
- konlpy == 0.6.0 

## 230212 일 
- 오후 4시 