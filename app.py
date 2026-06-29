import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# 페이지 설정
st.set_page_config(page_title="AI 글자 성적 예측기", layout="wide")
st.title("🎯 글자 성적(등급) 예측 시스템")

# 1. 모델 로드 (서버 직배송 방식 - 용량 제한 완벽 우회)
@st.cache_resource
def load_model_artifacts():
    model_path = 'font_grade_model.pkl'
    
    # 깃허브 업로드 실패 문제를 해결하기 위해 안전한 외부 클라우드 백업 주소에서 직접 다운로드합니다.
    remote_url = 'https://huggingface.co/datasets/rlatldn/backup/resolve/main/font_grade_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner("⏳ 대용량 AI 모델 파일을 서버에서 불러오는 중입니다... (최초 1회만 진행)"):
            try:
                # 데이터 유실 없이 완벽하게 바이너리 모델을 다운로드
                urllib.request.urlretrieve(remote_url, model_path)
            except Exception as e:
                st.error(f"❌ 모델 원격 다운로드 실패. 네트워크 상태를 확인하세요. 에러: {e}")
                return None
                
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ 가져온 모델 파일을 읽는 중 오류가 발생했습니다. (파일 손상 가능성): {e}")
        return None

artifacts = load_model_artifacts()
if artifacts:
    model = artifacts['model']
    le = artifacts['le']
    feature_columns = artifacts['feature_columns']
else:
    st.stop()

# 2. 피처 엔지니어링 함수
def engineer_features(df_input):
    X_raw = df_input.copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    X_eng = pd.DataFrame()
    eps = 1e-6
    
    target_cols = [
        '총 글자 수 (개)', '평균 글자 가로 크기 (px)', '글자 가로 크기 표준편차', 
        '평균 글자 세로 크기 (px)', '글자 세로 크기 표준편차', '평균 글자 가로/세로 비율', 
        '글자 비율 표준편차', '평균 글자 간 간격 (px)', '글자 간 간격 표준편차', 
        '평균 순수 글자 면적 (px2)', '글자 면적 표준편차', '평균 글자 범위 (Extent)', 
        '평균 글자 밀집도 (Solidity)', '글자 상단 정렬 표준편차', '글자 중심 세로 표준편차', 
        '평균 획 두께 추정 (px)', '획 두께 표준편차'
    ]
    
    for col in target_cols:
        if col in X_raw.columns:
            X_eng[col] = X_raw[col]
        else:
            X_eng[col] = 0
            
    X_eng['가로크기_CV']      = X_eng['글자 가로 크기 표준편차']     / (X_eng['평균 글자 가로 크기 (px)']     + eps)
    X_eng['세로크기_CV']      = X_eng['글자 세로 크기 표준편차']     / (X_eng['평균 글자 세로 크기 (px)']     + eps)
    X_eng['면적_CV']          = X_eng['글자 면적 표준편차']          / (X_eng['평균 순수 글자 면적 (px2)']    + eps)
    X_eng['간격_CV']          = X_eng['글자 간 간격 표준편차']       / (X_eng['평균 글자 간 간격 (px)']       + eps)
    X_eng['글자수_세로_비율'] = X_eng['총 글자 수 (개)']             / (X_eng['평균 글자 세로 크기 (px)']     + eps)
    X_eng['충실도']           = X_eng['평균 글자 밀집도 (Solidity)'] * X_eng['평균 글자 범위 (Extent)']
    X_eng['가로세로_비율']    = X_eng['평균 글자 가로 크기 (px)']    / (X_eng['평균 글자 세로 크기 (px)']     + eps)
    X_eng['면적대비_간격']    = X_eng['평균 글자 간 간격 (px)']       / (X_eng['평균 순수 글자 면적 (px2)']    + eps)
    X_eng['불규칙성']         = (X_eng['가로크기_CV'] + X_eng['세로크기_CV'] + X_eng['면적_CV']) / 3
    
    for c in feature_columns:
        if c not in X_eng.columns:
            X_eng[c] = 0
            
    return X_eng[feature_columns]

# 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["직접 입력", "구글 스프레드시트 불러오기"])

if menu == "직접 입력":
    st.subheader("📝 데이터 직접 입력")
    col1, col2 = st.columns(2)
    with col1:
        total_chars = st.number_input("총 글자 수 (개)", value=100)
        w_mean = st.number_input("평균 글자 가로 크기 (px)", value=15.0)
        w_std = st.number_input("글자 가로 크기 표준편차", value=1.2)
        h_mean = st.number_input("평균 글자 세로 크기 (px)", value=16.0)
        h_std = st.number_input("글자 세로 크기 표준편차", value=1.5)
        ratio_mean = st.number_input("평균 글자 가로/세로 비율", value=0.9)
        ratio_std = st.number_input("글자 비율 표준편차", value=0.1)
        gap_mean = st.number_input("평균 글자 간 간격 (px)", value=5.0)
        gap_std = st.number_input("글자 간 간격 표준편차", value=0.5)
    with col2:
        area_mean = st.number_input("평균 순수 글자 면적 (px2)", value=200.0)
        area_std = st.number_input("글자 면적 표준편차", value=15.0)
        extent = st.number_input("평균 글자 범위 (Extent)", value=0.75)
        solidity = st.number_input("평균 글자 밀집도 (Solidity)", value=0.85)
        align_std = st.number_input("글자 상단 정렬 표준편차", value=20.0)
        v_center_std = st.number_input("글자 중심 세로 표준편차", value=20.0)
        thick_mean = st.number_input("평균 획 두께 추정 (px)", value=15.0)
        thick_std = st.number_input("획 두께 표준편차", value=3.0)

    if st.button("🚀 예측하기"):
        input_dict = {
            '총 글자 수 (개)': [total_chars], '평균 글자 가로 크기 (px)': [w_mean], '글자 가로 크기 표준편차': [w_std], 
            '평균 글자 세로 크기 (px)': [h_mean], '글자 세로 크기 표준편차': [h_std], '평균 글자 가로/세로 비율': [ratio_mean], 
            '글자 비율 표준편차': [ratio_std], '평균 글자 간 간격 (px)': [gap_mean], '글자 간 간격 표준편차': [gap_std], 
            '평균 순수 글자 면적 (px2)': [area_mean], '글자 면적 표준편차': [area_std], '평균 글자 범위 (Extent)': [extent], 
            '평균 글자 밀집도 (Solidity)': [solidity], '글자 상단 정렬 표준편차': [align_std], '글자 중심 세로 표준편차': [v_center_std], 
            '평균 획 두께 추정 (px)': [thick_mean], '획 두께 표준편차': [thick_std]
        }
        X_p = engineer_features(pd.DataFrame(input_dict))
        res = le.inverse_transform(model.predict(X_p))[0]
        st.success(f"### 결과: **{res} 등급**")

elif menu == "구글 스프레드시트 불러오기":
    st.subheader("📂 구글 스프레드시트 데이터 예측")
    st.info("💡 주의: 스프레드시트는 구글 권한 설정에서 '링크가 있는 모든 사용자에게 공개(뷰어 혹은 편집자)' 상태여야 읽어올 수 있습니다.")
    sheet_url = st.text_input("스프레드시트 URL을 입력하세요:")
    
    if sheet_url:
        try:
            if "docs.google.com/spreadsheets" in sheet_url:
                sheet_id = sheet_url.split("/d/")[1].split("/")[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
                data = pd.read_csv(csv_url)
            else:
                st.error("올바른 구글 스프레드시트 URL 형식이 아닙니다.")
                st.stop()
                
            st.write(f"✅ 데이터를 성공적으로 불러왔습니다. (총 {len(data)}개의 행이 발견됨)")
            st.dataframe(data.head(5))
            
            row_idx = st.number_input(f"예측을 가동할 행 번호를 입력하세요 (0 ~ {len(data)-1}번 행 중 선택)", 
                                      min_value=0, max_value=len(data)-1, value=0)
            
            selected_row = data.iloc[[row_idx]]
            st.write(f"🔍 선택된 {row_idx}번 행의 수치 데이터:")
            st.dataframe(selected_row.drop(columns=['성적', '파일명'], errors='ignore'))
            
            if st.button("🔮 이 행의 등급 예측하기"):
                X_p = engineer_features(selected_row)
                res = le.inverse_transform(model.predict(X_p))[0]
                st.balloons()
                st.success(f"### AI 예측 결과: 해당 데이터는 **{res} 등급**으로 예측됩니다.")
                
        except Exception as e:
            st.error(f"시트를 분석하는 과정에서 에러가 발생했습니다. 링크 공유 권한 설정을 확인하세요. \n\n 에러내용: {e}")
