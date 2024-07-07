import pandas as pd

# CSV 파일 읽기
file_path = '../data/test_feature.csv'
df = pd.read_csv(file_path)

# 짝수 행 삭제
df_odd_rows = df.iloc[::2]

# 결과를 새로운 CSV 파일로 저장
df_odd_rows.to_csv('filtered_test.csv', index=False)
