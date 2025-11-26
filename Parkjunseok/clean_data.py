import pandas as pd
import numpy as np
import os 

# ---------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------
# [경로 수정] data 폴더 내부의 파일 읽기
file_path = 'data/tmdb_combined_10k.csv'

print(f"📂 '{file_path}' 데이터를 불러오는 중...")
if not os.path.exists(file_path):
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    print("load_data.py를 먼저 실행해 주세요.")
    exit()

df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. OTT 제공사(Providers) 전처리
# ---------------------------------------------------------
# (1) 결측치 처리
df['providers'] = df['providers'].fillna('기타')

# (2) Netflix 통합 및 데이터 정리 함수
def clean_providers(provider_str):
    if provider_str == '기타':
        return provider_str
        
    providers = [p.strip() for p in provider_str.split(',')]
    new_providers = set()
    
    for p in providers:
        if p == 'Netflix Standard with Ads':
            new_providers.add('Netflix')
        else:
            new_providers.add(p)
            
    return ', '.join(sorted(list(new_providers)))

df['providers'] = df['providers'].apply(clean_providers)

# ---------------------------------------------------------
# 3. 장르(Genres) 전처리
# ---------------------------------------------------------
df['genres'] = df['genres'].fillna('')

# 알려진 장르 수동 매핑
known_genre_map = {
    'Menekşe Gözler': '로맨스, 뮤지컬, 드라마',
    'くノ一忍法帖 自来也秘抄': '액션, 판타지',
    'The World Famous Musical Comedy Artists Seymour Hicks and Ellaline Terriss in a Selection of Their Dances': '다큐멘터리',
    'Two-Eleven': '드라마',
    'Delectable Destinations': '다큐멘터리',
    'Effetto Olmi': '다큐멘터리',
    'Manila Scream': '공포',
    'Nahual': '공포, 스릴러',
    'Illusion': '드라마',
    'Looping': '드라마',
    'Janata Bar': '드라마, 범죄',
    'Khanna & Iyer': '코미디, 로맨스',
    'Aldrin űropera': 'TV 영화',
    'Mis abuelitas... no más!': '코미디'
}

for title, genre in known_genre_map.items():
    df.loc[(df['title'] == title) & (df['genres'] == ''), 'genres'] = genre

# 나머지는 '기타' 처리
df.loc[df['genres'] == '', 'genres'] = '기타'

# ---------------------------------------------------------
# 4. [NEW] 가중 평점(Weighted Score) 추가
# ---------------------------------------------------------
print("⚖️ 가중 평점(Weighted Score) 계산 중...")

# (1) 전체 평균 평점 (C)
C = df['vote_average'].mean()

# (2) 최소 투표수 기준 (m) - 상위 10% 기준 적용
# (이 점수는 '인기 추천작' 랭킹용으로 주로 쓰이므로 높은 기준 적용)
m = df['vote_count'].quantile(0.90)

# (3) 가중 평점 계산 함수
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

# (4) 컬럼 생성
df['weighted_score'] = df.apply(weighted_rating, axis=1)

# ---------------------------------------------------------
# 5. 파일로 저장하기
# ---------------------------------------------------------
# [경로 수정] data 폴더 내부에 저장
output_filename = 'data/tmdb_cleaned.csv'

# 한글 깨짐 방지를 위해 utf-8-sig 사용
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n" + "="*40)
print(f"✅ 정제 및 가중 평점 계산 완료!")
print(f"💾 '{output_filename}' 파일이 생성되었습니다.")
print("="*40)

# 결과 확인
print(f"\n[생성된 파일 정보]")
print(f"- 저장 위치: {os.path.abspath(output_filename)}")
print(f"- 총 데이터 개수: {len(df)}")
print(f"- 가중 평점(weighted_score) 컬럼 추가됨 ✅")
print(f"- 제공사 '기타' 개수: {len(df[df['providers'] == '기타'])}")