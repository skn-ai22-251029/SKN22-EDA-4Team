import requests
import pandas as pd
from tqdm import tqdm
import time

# API 설정 (사용 시 본인의 키로 변경 필요)
API_KEY = "YOUR_API_KEY_HERE"  # 👈 여기에 TMDB API 키를 입력하세요!
BASE = "https://api.themoviedb.org/3"

# 수집할 타겟 설정 (영화, TV쇼)
targets = ["movie", "tv"]
records = []

print("데이터 수집을 시작합니다... (Movie + TV Show)")

for content_type in targets:
    print(f"\n[{content_type.upper()}] 수집 시작")
    
    # --- 1) Popular ID 수집 ---
    ids = set()
    # 250페이지 * 20개 * 2종류 = 약 10,000개 데이터 목표
    # 시간 너무 오래 걸리면 range(1, 251)을 range(1, 51) 정도로 줄이세요.
    for page in tqdm(range(1, 251), desc=f"Collecting {content_type} IDs"):
        url = f"{BASE}/{content_type}/popular?api_key={API_KEY}&language=ko-KR&page={page}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    ids.add(item["id"])
            else:
                print(f"Error on page {page}: {response.status_code}")
        except Exception as e:
            print(f"Exception on page {page}: {e}")
            
        time.sleep(0.05) # 속도를 위해 딜레이 약간 줄임

    id_list = list(ids)
    print(f"👉 {content_type} ID {len(id_list)}개 확보 완료. 상세 정보 수집 시작...")

    # --- 2) 상세 정보 + OTT 불러오기 ---
    for content_id in tqdm(id_list, desc=f"Fetching {content_type} Details"):
        try:
            # 상세 정보 URL (영화/TV 구분)
            detail_url = f"{BASE}/{content_type}/{content_id}?api_key={API_KEY}&language=ko-KR"
            providers_url = f"{BASE}/{content_type}/{content_id}/watch/providers?api_key={API_KEY}"

            detail = requests.get(detail_url).json()
            providers = requests.get(providers_url).json()

            # 1. 장르 추출
            genres = [g["name"] for g in detail.get("genres", [])]

            # 2. OTT 정보 추출 (한국 기준)
            kr_provider = providers.get("results", {}).get("KR", {})
            flatrate = kr_provider.get("flatrate", []) or [] # 정액제(구독)만 추출
            otts = [p["provider_name"] for p in flatrate]

            # 3. 제목 필드 통일 (영화: title, TV: name)
            # TV쇼는 title이 없고 name이 있습니다.
            title = detail.get("title") if content_type == "movie" else detail.get("name")
            
            # 4. 투표수/평점 (인기도 분석용)
            vote_count = detail.get("vote_count", 0)
            vote_average = detail.get("vote_average", 0)

            records.append({
                "id": content_id,
                "title": title,
                "type": content_type, # movie 인지 tv 인지 구분
                "genres": ",".join(genres), # 리스트를 문자열로 변환 (csv 저장용)
                "providers": ",".join(otts), # 리스트를 문자열로 변환
                "vote_count": vote_count,
                "vote_average": vote_average
            })

        except Exception as e:
            # print(f"Error on ID {content_id}: {e}") # 에러 로그가 너무 많으면 주석 처리
            continue

        time.sleep(0.05) # 차단 방지

# --- 3) DataFrame 구성 및 저장 ---
df = pd.DataFrame(records)

print("\n==============================")
print(df.head())
print(f"최종 데이터 수: {len(df)}개")
print(f"   - Movie: {len(df[df['type']=='movie'])}")
print(f"   - TV:    {len(df[df['type']=='tv'])}")

# CSV 저장
file_name = "tmdb_combined_10k.csv"
df.to_csv(file_name, index=False, encoding="utf-8-sig")
print(f"'{file_name}' 파일로 저장 완료!")
