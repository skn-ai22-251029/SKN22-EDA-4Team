import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import numpy as np

# ---------------------------------------------------------
# 1. 환경 설정
# ---------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False 

IMAGE_DIR = 'images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
file_path = 'data/tmdb_cleaned.csv'
if not os.path.exists(file_path):
    print("❌ 데이터 파일이 없습니다.")
    exit()

df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 3. 추천 알고리즘 (중위권 제외로 확실한 분리)
# ---------------------------------------------------------
C = df['vote_average'].mean()

# [인기작] 상위 10% (약 3,000표 이상)
m_popular = df['vote_count'].quantile(0.90) 

# [숨은 명작] 상위 25% 미만 (약 450표 이하) ~ 50표 이상
# (중간인 450표 ~ 3,000표 구간은 '일반 작품'으로 분류하여 제외함)
m_hidden_max = df['vote_count'].quantile(0.75) 
m_hidden_min = 50 

print(f"\n📊 [그룹핑 기준값 설정]")
print(f" - 인기작 기준 (Popular): {m_popular:.0f}표 이상")
print(f" - 숨은 명작 기준 (Hidden): {m_hidden_min}표 ~ {m_hidden_max:.0f}표 (확실히 숨겨진 작품만)")

# 가중 평점 계산
def weighted_rating(x, m=m_popular, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

df['weighted_score'] = df.apply(weighted_rating, axis=1)

# (1) 인기 추천작
popular_recs = df[
    (df['vote_count'] >= m_popular) & 
    (df['vote_average'] >= C)
].sort_values('weighted_score', ascending=False)

# (2) 숨은 명작
hidden_gems = df[
    (df['vote_count'] < m_hidden_max) &    # 450표 미만만 인정 (중위권 제외)
    (df['vote_count'] >= m_hidden_min) &   
    (df['vote_average'] >= 7.0)            
].sort_values('vote_average', ascending=False)

print("\n🎬 [인기 추천 Top 5]"); print(popular_recs[['title', 'vote_average']].head(5))
print("\n💎 [숨은 명작 Top 5]"); print(hidden_gems[['title', 'vote_average']].head(5))

# =========================================================
# 4. 시각화 (Total 5 Charts)
# =========================================================

# (1) 장르별 소비량
print("\n🎨 1. 장르별 소비량...")
df['genres_split'] = df['genres'].str.split(',')
df_exploded = df.explode('genres_split')
df_exploded['genres_split'] = df_exploded['genres_split'].str.strip()

plt.figure(figsize=(14, 7))
genre_counts = df_exploded['genres_split'].value_counts().head(15)
ax = sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='mako', legend=False)
for i, v in enumerate(genre_counts.values):
    ax.text(v + 10, i, f"{v:,}편", va='center', fontsize=10, fontweight='bold', color='black')
plt.title('장르별 작품 수 Top 15 (소비량)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '1_genre_ranking.png'), dpi=150)

# (2) 그룹별 특성 비교 (Box Plot)
print("🎨 2. 그룹별 특성 비교 박스플롯...")
popular_recs['Group'] = '인기 추천작 (Top 10%↑)'
hidden_gems['Group'] = '숨은 명작 (Top 25%↓)'
comparison_df = pd.concat([popular_recs, hidden_gems])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=comparison_df, x='Group', y='vote_average', palette=['#5DADE2', '#EC7063'], width=0.5, ax=axes[0])
axes[0].set_title('작품성 비교 (평점)', fontsize=14, fontweight='bold')
sns.boxplot(data=comparison_df, x='Group', y='vote_count', palette=['#5DADE2', '#EC7063'], width=0.5, ax=axes[1])
axes[1].set_title('대중성 비교 (투표수)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
plt.suptitle('인기작 vs 숨은 명작 : 그룹별 특성 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '2_group_comparison_boxplot.png'), dpi=150)

# (2-Bonus) 숨은 명작 포지셔닝 맵 (Scatter) - [영역 분리 확인용]
print("🎨 2-Bonus. 포지셔닝 맵 (중위권 공백 확인)...")
plt.figure(figsize=(14, 8))
# 배경 (일반 작품 - 중위권 포함)
plt.scatter(df['vote_count'], df['vote_average'], alpha=0.15, color='#CCCCCC', label='일반 작품', s=15, zorder=1)
# 인기작 (오른쪽 끝)
plt.scatter(popular_recs['vote_count'], popular_recs['vote_average'], alpha=0.6, color='#5DADE2', label='인기 명작', s=40, zorder=2)
# 숨은 명작 (왼쪽 끝)
plt.scatter(hidden_gems['vote_count'], hidden_gems['vote_average'], alpha=0.8, color='#EC7063', label='숨은 명작', s=60, zorder=3, edgecolors='none')

# 기준선
plt.axvline(x=m_popular, color='b', linestyle='--', label='인기작 기준')
plt.axvline(x=m_hidden_max, color='r', linestyle='--', label='숨은 명작 상한선')

plt.xscale('log')
plt.title('투표수 vs 평점 분포 (중위권 제외로 확실한 분리)', fontsize=16, fontweight='bold')
plt.xlabel('투표수 (Log Scale)')
plt.ylabel('평점')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '2_positioning_map.png'), dpi=150)


# (3) OTT 히트맵
print("🎨 3. OTT 히트맵...")
top_providers = df['providers'].str.split(',').explode().str.strip().value_counts().head(7).index.tolist()
if '기타' in top_providers: top_providers.remove('기타')
top_genres = df['genres'].str.split(',').explode().str.strip().value_counts().head(10).index.tolist()
if '기타' in top_genres: top_genres.remove('기타')
corr_df = pd.DataFrame()
for provider in top_providers:
    corr_df[provider] = df['providers'].apply(lambda x: 1 if provider in [p.strip() for p in x.split(',')] else 0)
for genre in top_genres:
    corr_df[genre] = df['genres'].apply(lambda x: 1 if genre in [g.strip() for g in x.split(',')] else 0)
target_corr = corr_df.corr().loc[top_providers, top_genres]
plt.figure(figsize=(12, 9))
sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, linewidths=1, linecolor='white', cbar_kws={'label': '상관계수'}, square=True)
plt.title('OTT 플랫폼별 장르 특화도', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '3_ott_genre_correlation.png'), dpi=150)

# (4) OTT 비율 비교 - [수정: 넷플릭스 누락 해결]
print("🎨 4. OTT 비율 비교...")
def get_distribution(target_df, column_name):
    split_data = target_df[column_name].str.split(',').explode().str.strip()
    split_data = split_data[split_data != '기타']
    return split_data.value_counts(normalize=True) * 100

pop_ott = get_distribution(popular_recs, 'providers')
hidden_ott = get_distribution(hidden_gems, 'providers')

# [핵심 수정] 두 그룹 합쳐서 '가장 많이 등장한' 상위 7개 OTT 선정 (알파벳순 X, 빈도순 O)
combined_counts = pop_ott.add(hidden_ott, fill_value=0)
top_otts = combined_counts.sort_values(ascending=False).head(7).index 

ott_comp = pd.DataFrame({'Popular': pop_ott, 'Hidden': hidden_ott}).loc[top_otts].fillna(0)
ott_comp.index.name = 'OTT_Platform'
ott_comp = ott_comp.reset_index().melt(id_vars='OTT_Platform')

plt.figure(figsize=(12, 7))
sns.barplot(data=ott_comp, x='OTT_Platform', y='value', hue='variable', palette={'Popular': '#4A90E2', 'Hidden': '#E74C3C'})
plt.title('인기작 vs 숨은 명작 OTT 보유 비율 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '4_ott_comparison_bar.png'), dpi=150)

# (5) 콘텐츠 유형 비교
print("🎨 5. 콘텐츠 유형 비교...")
def get_type_ratio(target_df):
    return target_df['type'].value_counts(normalize=True) * 100
type_df = pd.DataFrame({'Popular': get_type_ratio(popular_recs), 'Hidden': get_type_ratio(hidden_gems)}).T
type_df.index = ['인기 추천작', '숨은 명작']
plt.figure(figsize=(10, 6))
type_df.plot(kind='barh', stacked=True, color=['#FFB3BA', '#BAE1FF'], figsize=(10, 6), width=0.6)
for n, x in enumerate([*type_df.index.values]):
    for (img, label) in zip(type_df.loc[x], type_df.columns):
        if img > 5:
            plt.text(type_df.loc[x].cumsum()[label] - (img / 2), n, f"{label.upper()}\n{img:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=12)
plt.title('그룹별 콘텐츠 유형 비중 (Movie vs TV)', fontsize=16, fontweight='bold')
plt.legend(title='Type', loc='upper right', bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, '5_type_comparison.png'), dpi=150)

print("\n" + "="*40)
print(f"✅ 최종 분석 완료! (넷플릭스 복구 & 그룹 분리)")
print(f"📁 결과물: {os.path.abspath(IMAGE_DIR)}")
print("="*40)