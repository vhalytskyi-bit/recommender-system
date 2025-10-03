"""
Веб-додаток для демонстрації рекомендаційної системи
Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Налаштування сторінки
st.set_page_config(
    page_title="Система рекомендацій фільмів",
    page_icon="🎬",
    layout="wide"
)

# Кешування даних
@st.cache_data
def load_data():
    """Завантажує дані"""
    ratings = pd.read_csv('data/processed/ratings.csv')
    movies = pd.read_csv('data/processed/movies.csv')
    users = pd.read_csv('data/processed/users.csv')
    return ratings, movies, users

@st.cache_data
def create_matrices(ratings):
    """Створює матрицю користувач-фільм"""
    matrix = ratings.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )
    return matrix

@st.cache_data
def compute_similarity(matrix):
    """Обчислює подібність між фільмами"""
    item_similarity = cosine_similarity(matrix.T)
    return item_similarity

# Item-based рекомендації
def get_item_based_recommendations(user_id, matrix, item_similarity, movies, n=10):
    """Генерує топ-N рекомендацій для користувача"""
    
    if user_id not in matrix.index:
        return None
    
    # Отримуємо оцінки користувача
    user_ratings = matrix.loc[user_id]
    
    # Фільми які користувач вже оцінив
    rated_items = user_ratings[user_ratings > 0].index.tolist()
    
    if len(rated_items) == 0:
        return None
    
    # Обчислюємо передбачені оцінки для всіх неоцінених фільмів
    predictions = {}
    
    for item_id in matrix.columns:
        if item_id in rated_items:
            continue
        
        # Знаходимо індекс фільму
        item_idx = matrix.columns.get_loc(item_id)
        
        # Подібності до всіх фільмів
        similarities = item_similarity[item_idx]
        
        # Фільтруємо тільки оцінені фільми
        rated_mask = user_ratings > 0
        similarities_rated = similarities[rated_mask.values]
        ratings_rated = user_ratings[rated_mask].values
        
        if len(similarities_rated) == 0:
            continue
        
        # Вибираємо топ-20 найсхожіших
        k = min(20, len(similarities_rated))
        top_k_idx = np.argsort(similarities_rated)[-k:]
        top_similarities = similarities_rated[top_k_idx]
        top_ratings = ratings_rated[top_k_idx]
        
        if top_similarities.sum() == 0:
            continue
        
        # Зважене середнє
        pred = np.dot(top_similarities, top_ratings) / top_similarities.sum()
        predictions[item_id] = pred
    
    # Сортуємо за передбаченою оцінкою
    top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Отримуємо інформацію про фільми
    recommendations = []
    for item_id, pred_rating in top_items:
        movie_info = movies[movies['item_id'] == item_id].iloc[0]
        
        # Жанри
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                      'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                      'Thriller', 'War', 'Western']
        
        genres = [col for col in genre_cols if movie_info[col] == 1]
        
        recommendations.append({
            'title': movie_info['title'],
            'predicted_rating': pred_rating,
            'genres': ', '.join(genres) if genres else 'Unknown'
        })
    
    return recommendations

# Отримання історії користувача
def get_user_history(user_id, ratings, movies, n=10):
    """Показує історію оцінок користувача"""
    user_ratings = ratings[ratings['user_id'] == user_id].copy()
    user_ratings = user_ratings.merge(movies[['item_id', 'title']], on='item_id')
    user_ratings = user_ratings.sort_values('rating', ascending=False).head(n)
    return user_ratings[['title', 'rating']]

# Головна функція
def main():
    st.title("🎬 Система рекомендацій фільмів")
    st.markdown("---")
    
    # Завантаження даних
    with st.spinner("Завантаження даних..."):
        ratings, movies, users = load_data()
        matrix = create_matrices(ratings)
        item_similarity = compute_similarity(matrix)
    
    # Sidebar
    st.sidebar.header("Налаштування")
    
    # Вибір користувача
    user_id = st.sidebar.selectbox(
        "Оберіть користувача:",
        options=sorted(matrix.index.tolist()),
        index=0
    )
    
    # Кількість рекомендацій
    n_recommendations = st.sidebar.slider(
        "Кількість рекомендацій:",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Інформація про користувача
    st.sidebar.markdown("---")
    st.sidebar.subheader("Інформація про користувача")
    
    user_info = users[users['user_id'] == user_id].iloc[0]
    num_ratings = len(ratings[ratings['user_id'] == user_id])
    avg_rating = ratings[ratings['user_id'] == user_id]['rating'].mean()
    
    st.sidebar.write(f"**ID:** {user_id}")
    st.sidebar.write(f"**Вік:** {user_info['age']}")
    st.sidebar.write(f"**Стать:** {'Чоловік' if user_info['gender'] == 'M' else 'Жінка'}")
    st.sidebar.write(f"**Професія:** {user_info['occupation']}")
    st.sidebar.write(f"**Кількість оцінок:** {num_ratings}")
    st.sidebar.write(f"**Середня оцінка:** {avg_rating:.2f}")
    
    # Основний контент
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Історія оцінок")
        st.write("Топ-10 найвище оцінених фільмів:")
        
        history = get_user_history(user_id, ratings, movies, n=10)
        
        if len(history) > 0:
            # Форматування таблиці
            history_display = history.copy()
            history_display['rating'] = history_display['rating'].apply(lambda x: '⭐' * int(x))
            
            st.dataframe(
                history_display,
                column_config={
                    "title": st.column_config.TextColumn("Назва фільму", width="large"),
                    "rating": st.column_config.TextColumn("Оцінка", width="small"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Користувач ще не оцінив жодного фільму")
    
    with col2:
        st.subheader("🎯 Рекомендації")
        
        with st.spinner("Генерація рекомендацій..."):
            recommendations = get_item_based_recommendations(
                user_id, matrix, item_similarity, movies, n=n_recommendations
            )
        
        if recommendations:
            st.write(f"Топ-{n_recommendations} рекомендованих фільмів:")
            
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"**{i}. {rec['title']}**")
                    
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.write(f"Оцінка: {rec['predicted_rating']:.2f} ⭐")
                    with col_b:
                        st.write(f"Жанри: {rec['genres']}")
                    
                    st.markdown("---")
        else:
            st.warning("Неможливо згенерувати рекомендації для цього користувача")
    
    # Статистика системи
    st.markdown("---")
    st.subheader("📈 Статистика системи")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Користувачів", f"{len(users):,}")
    
    with col2:
        st.metric("Фільмів", f"{len(movies):,}")
    
    with col3:
        st.metric("Оцінок", f"{len(ratings):,}")
    
    with col4:
        sparsity = (1 - len(ratings) / (len(users) * len(movies))) * 100
        st.metric("Розрідженість", f"{sparsity:.1f}%")
    
    # Інформація про модель
    st.markdown("---")
    with st.expander("ℹ️ Про систему"):
        st.markdown("""
        ### Про рекомендаційну систему
        
        Ця система використовує **Item-based Collaborative Filtering** для генерації персоналізованих рекомендацій фільмів.
        
        **Як це працює:**
        1. Система аналізує схожість між фільмами на основі того, як їх оцінювали користувачі
        2. Для кожного користувача система знаходить фільми схожі на ті, що він оцінив високо
        3. Генерується список рекомендацій з передбаченими оцінками
        
        **Датасет:** MovieLens 100K
        - 943 користувачі
        - 1,682 фільми
        - 100,000 оцінок
        
        **Точність моделі:**
        - RMSE: 0.9714
        - MAE: 0.7591
        
        **Технології:** Python, Pandas, Scikit-learn, Streamlit
        """)

if __name__ == "__main__":
    main()