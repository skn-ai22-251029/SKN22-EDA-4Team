import pandas as pd
import numpy as np
import os 

# 1. 데이터 불러오기
file_path = 'data/tmdb_combined_10k.csv'

print(f"'{file_path}' 데이터를 불러오는 중...")
df = pd.read_csv(file_path)

# 2. OTT 제공사(Providers) 전처리
# (1) 결측치 처리: 어딘가엔 있을 테니 '기타'로 설정
df['providers'] = df['providers'].fillna('기타')

# (2) Netflix 통합 및 데이터 정리 함수
def clean_providers(provider_str):
    # '기타'인 경우 그대로 반환
    if provider_str == '기타':
        return provider_str
        
    # 콤마로 분리
    providers = [p.strip() for p in provider_str.split(',')]
    new_providers = set()
    
    for p in providers:
        # 광고형 넷플릭스를 일반 넷플릭스로 통합
        if p == 'Netflix Standard with Ads':
            new_providers.add('Netflix')
        else:
            new_providers.add(p)
            
    # 정렬하여 다시 문자열로 합치기
    return ', '.join(sorted(list(new_providers)))

# 함수 적용
df['providers'] = df['providers'].apply(clean_providers)

# 3. 장르(Genres) 전처리
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

# 나머지 장르 미상도 '기타'로 처리
df.loc[df['genres'] == '', 'genres'] = '기타'

# 4. 파일로 저장하기
output_filename = 'tmdb_cleaned.csv'

# 한글 깨짐 방지를 위해 utf-8-sig 사용
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n" + "="*40)
print(f"정제 작업 완료")
print(f"'{output_filename}' 파일이 생성되었습니다.")
print("="*40)

# 결과 확인
print(f"\n[생성된 파일 정보]")
print(f"- 저장 위치: {os.path.abspath(output_filename)}")
print(f"- 총 데이터 개수: {len(df)}")
print(f"- 제공사 '기타' 개수: {len(df[df['providers'] == '기타'])}")