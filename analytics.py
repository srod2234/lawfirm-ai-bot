import streamlit as st
from sqlmodel import Session, select, func
from models import engine, User, Document, ChatMessage
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def show_dashboard():
    st.title("📊 Analytics Dashboard")

    # Sidebar controls
    days = st.sidebar.slider("Days to show", min_value=7, max_value=90, value=30)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    # 1) Documents per user
    st.subheader("Documents per User")
    with Session(engine) as sess:
        rows = sess.exec(
            select(User.username, func.count(Document.id))
            .join(Document, Document.owner_id == User.id)
            .group_by(User.username)
        ).all()
    if rows:
        users, counts = zip(*rows)
        fig, ax = plt.subplots()
        ax.bar(users, counts)
        ax.set_ylabel("Documents")
        ax.set_xticklabels(users, rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.write("No documents to display.")

    # 2) Q&A counts over time
    st.subheader(f"Q&A Count (Last {days} Days)")
    with Session(engine) as sess:
        qa = sess.exec(
            select(
                func.date(ChatMessage.timestamp),
                func.count(ChatMessage.id)
            )
            .where(ChatMessage.timestamp >= start_date)
            .group_by(func.date(ChatMessage.timestamp))
            .order_by(func.date(ChatMessage.timestamp))
        ).all()
    if qa:
        dates, qcounts = zip(*qa)
        fig2, ax2 = plt.subplots()
        ax2.plot(dates, qcounts)
        ax2.set_ylabel("Q&A Count")
        ax2.set_xticklabels(dates, rotation=45, ha="right")
        st.pyplot(fig2)
    else:
        st.write("No chat activity in this period.")

    # 3) Uploads over time
    st.subheader(f"Uploads (Last {days} Days)")
    with Session(engine) as sess:
        up = sess.exec(
            select(
                func.date(Document.uploaded_at),
                func.count(Document.id)
            )
            .where(Document.uploaded_at >= start_date)
            .group_by(func.date(Document.uploaded_at))
            .order_by(func.date(Document.uploaded_at))
        ).all()
    if up:
        udates, ucounts = zip(*up)
        fig3, ax3 = plt.subplots()
        ax3.bar(udates, ucounts)
        ax3.set_ylabel("Uploads")
        ax3.set_xticklabels(udates, rotation=45, ha="right")
        st.pyplot(fig3)
    else:
        st.write("No uploads in this period.")
