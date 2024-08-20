import streamlit as st

class MultiPage:
    def __init__(self, app_name):
        st.set_page_config(page_title=app_name)
        self.pages = []

    def add_page(self, title, func):
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        page = st.sidebar.selectbox(
            'Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()
