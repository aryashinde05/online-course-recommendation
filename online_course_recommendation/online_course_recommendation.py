import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('udemy_course_data.csv')  # Make sure to adjust the path to your dataset

# Page Configuration
st.set_page_config(page_title='Online Course Recommendation System', layout='wide', page_icon='📚')

# Sidebar - About the App
st.sidebar.title("💡About the App")
st.sidebar.write(""" 
This app recommends online courses based on user preferences. 
Please select your interests and find suitable courses to enhance your skills!
""")

# Banner
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>📚 Online Course Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #3F51B5;'>Find Your Perfect Course!</h2>", unsafe_allow_html=True)

# Add some spacing
st.write("<br>", unsafe_allow_html=True)

# Convert content_duration to hours
def convert_duration(duration):
    try:
        if isinstance(duration, str):
            duration = duration.strip().lower()
            if 'hour' in duration:
                return float(duration.split()[0])
            elif 'min' in duration:  # If the duration is in minutes, convert it to hours
                return float(duration.split()[0]) / 60
            else:
                return float('nan')  # Return NaN for unexpected formats
        else:
            return float('nan')  # Return NaN for unexpected types
    except Exception as e:
        return float('nan')  # Handle errors gracefully

df['content_duration'] = df['content_duration'].apply(convert_duration)

# Create mappings for subjects and levels
subject_mapping = {subject: idx for idx, subject in enumerate(df['subject'].unique())}
level_mapping = {level: idx for idx, level in enumerate(df['level'].unique())}

df['subject_encoded'] = df['subject'].map(subject_mapping)
df['level_encoded'] = df['level'].map(level_mapping)
df['is_paid'] = df['is_paid'].astype(int)

# Prepare the dataset for training
X = df[['subject_encoded', 'level_encoded', 'price', 'num_lectures', 'content_duration', 'year', 'is_paid']]
y = df['subject_encoded']  # Predicting subject as a class

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Feature Selection
st.header("Select Your Preferences")
col1, col2 = st.columns(2)

with col1:
    selected_subject = st.selectbox('📘 Select a Subject', list(subject_mapping.keys()))
    selected_level = st.selectbox('📈 Select Course Level', list(level_mapping.keys()))
    selected_price = st.slider('💵 Maximum Price', min_value=0, max_value=int(df['price'].max()), value=100)
    selected_num_lectures = st.slider('📊 Maximum Number of Lectures', min_value=1, max_value=int(df['num_lectures'].max()), value=10)

with col2:
    selected_duration = st.slider('⏱️ Maximum Duration (in hours)', min_value=0, max_value=int(df['content_duration'].max()), value=5)
    selected_year = st.selectbox('📅 Year of Publication', sorted(df['year'].unique(), reverse=True))
    selected_paid_status = st.selectbox('💰 Course Price Type', ['Paid', 'Free'])

# Predict using Decision Tree Classifier
input_data = {
    'subject_encoded': [subject_mapping[selected_subject]],
    'level_encoded': [level_mapping[selected_level]],
    'price': [selected_price],
    'num_lectures': [selected_num_lectures],
    'content_duration': [selected_duration],
    'year': [selected_year],
    'is_paid': [1 if selected_paid_status == 'Paid' else 0]
}

input_df = pd.DataFrame(input_data)

# Prediction based on user inputs using Decision Tree
predicted_subject_encoded_tree = decision_tree_model.predict(input_df)[0]
predicted_subject_name_tree = list(subject_mapping.keys())[list(subject_mapping.values()).index(predicted_subject_encoded_tree)]

# Prediction based on user inputs using Random Forest
predicted_subject_encoded_forest = random_forest_model.predict(input_df)[0]
predicted_subject_name_forest = list(subject_mapping.keys())[list(subject_mapping.values()).index(predicted_subject_encoded_forest)]

# Filter Courses Based on User Input
filtered_courses = df[
    (df['subject_encoded'] == subject_mapping[selected_subject]) & 
    (df['level_encoded'] == level_mapping[selected_level]) & 
    (df['price'] <= selected_price) & 
    (df['num_lectures'] <= selected_num_lectures) & 
    (df['content_duration'] <= selected_duration) & 
    (df['year'] == selected_year) & 
    (df['is_paid'] == (1 if selected_paid_status == 'Paid' else 0))
]

# Display Recommended Courses from Decision Tree
if not filtered_courses.empty:
    st.subheader("Top 3 Most Recommended Courses")
    
    sorted_courses_tree = filtered_courses.sort_values(by='price', ascending=True)
    top_courses_tree = sorted_courses_tree.head(3)
    st.dataframe(top_courses_tree[['course_title', 'url', 'price', 'num_lectures']])

    # Suggest related courses using Random Forest
    related_courses = df[df['subject_encoded'] == predicted_subject_encoded_forest].head(10)
    
    if not related_courses.empty:
        st.subheader("You Might Also Like")
        st.dataframe(related_courses[['course_title', 'url', 'price', 'num_lectures']])
    else:
        st.warning("No related courses found based on Random Forest recommendations.")
else:
    st.warning("No courses found matching your preferences.")

# Data Distribution Visualization
if st.checkbox('Show Data Distribution'):
    st.subheader("Data Distribution")

    # Visualize distribution of subjects
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='subject', palette='Set2')
    plt.title('Distribution of Subjects', fontsize=16)
    plt.xticks(rotation=45)
    plt.xlabel('Subjects', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    st.pyplot(plt)

    # Visualize distribution of content duration
    plt.figure(figsize=(10, 6))
    sns.histplot(df['content_duration'].dropna(), bins=20, kde=True, color='lightblue')
    plt.title('Distribution of Course Duration', fontsize=16)
    plt.xlabel('Duration (in hours)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    st.pyplot(plt)

    # Visualize number of lectures
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_lectures'], bins=20, kde=True, color='lightgreen')
    plt.title('Distribution of Number of Lectures', fontsize=16)
    plt.xlabel('Number of Lectures', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    st.pyplot(plt)

    # Additional distribution: Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=20, kde=True, color='lightcoral')
    plt.title('Distribution of Course Price', fontsize=16)
    plt.xlabel('Price', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    st.pyplot(plt)

    # Correlation Heatmap
    if st.checkbox('Show Correlation Heatmap'):
        plt.figure(figsize=(10, 6))
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()  # Calculate correlation only on numeric columns
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap', fontsize=16)
        st.pyplot(plt)

# Additional Data Visualization
if st.checkbox('Show All Courses'):
    st.subheader('All Available Courses')
    st.dataframe(df[['course_title', 'subject', 'level', 'price', 'num_lectures']])

# Footer with Styling
st.markdown(""" 
    <hr style='border:1px solid #eee'>
    <div style='text-align:center; background-color: #f9f9f9; padding: 10px;'>
    </div>
""", unsafe_allow_html=True)
