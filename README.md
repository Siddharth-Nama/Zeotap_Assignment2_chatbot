# Chatbot with Gemini API

## 🔹 Overview
This project implements a **Django-based AI-powered chatbot** that answers user queries by dynamically accessing documentation URLs of various **Customer Data Platforms (CDPs)** and integrating with the **Google Gemini API** for general inquiries. The chatbot does not rely on local storage of documentation, ensuring that responses are always **based on the latest online data**.

### 🌍 Supported CDPs:
- **Segment** → [Documentation](https://segment.com/docs/)
- **mParticle** → [Documentation](https://docs.mparticle.com/)
- **Lytics** → [Documentation](https://docs.lytics.com/)
- **Zeotap** → [Documentation](https://docs.zeotap.com/)

If a question is not specific to a CDP, the chatbot leverages **Google Gemini API** to provide a relevant response.

---

## 🚀 Key Features
✅ **Real-Time Web Scraping** – Fetches updated documentation content dynamically.  
✅ **Google Gemini API Integration** – Answers general queries beyond CDPs.  
✅ **Natural Language Processing (NLP)** – Uses sentence embeddings for intelligent query matching.  
✅ **Smart Formatting** – Understands structured data from CDP documentation.  
✅ **User-Friendly Interface** – Simple chat UI built with Django templates.  
✅ **Undo/Redo Stack** – Maintains conversation history for better usability.  

---

## ⚙️ Project Architecture
```
cdp_chatbot/        # Django Project Folder
│── chatbot/        # Django App
│   │── views.py    # Handles request-response cycle
│   │── utils.py    # Implements web scraping, embeddings & processing
│   │── urls.py     # URL routing
│   └── templates/  # HTML templates for chat UI
│── static/         # CSS files for styling
│── manage.py       # Django management script
└── .env            # API keys and environment variables
```

---

## 💡 Tech Stack & Dependencies
| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Django** | Web framework for chatbot backend |
| **Beautiful Soup 4** | Web scraping for extracting documentation text |
| **Requests** | Fetching online documentation dynamically |
| **Sentence Transformers** | NLP embeddings for semantic similarity |
| **Scikit-Learn** | Cosine similarity computation for search ranking |
| **NumPy** | Efficient matrix computations |
| **Google Generative AI** | Integration with Gemini API |
| **Gunicorn** | Deployment-ready application server |
| **python-dotenv** | Secure API key management |

---

## 🔧 Setup & Installation
### 1️⃣ Clone Repository
```bash
git clone <https://github.com/Siddharth-Nama/Zeotap_Assignment2_chatbot>
cd <zeotap_chatbot>
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment
- **Windows (cmd.exe)**: `venv\Scripts\activate`
- **Windows (PowerShell)**: `venv\Scripts\activate`
- **Linux/macOS**: `source venv/bin/activate`

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
> **Dependencies Installed:** Django, BeautifulSoup4, Requests, Sentence-Transformers, Scikit-Learn, NumPy, google-generativeai, python-dotenv, Gunicorn

### 5️⃣ Configure API Keys
- **Create a `.env` file** in the root directory.
- Add the following:
```ini
GOOGLE_API_KEY="your-google-gemini-api-key"
SECRET_KEY="your-django-secret-key"
```

### 6️⃣ Apply Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 7️⃣ Run the Development Server
```bash
python manage.py runserver
```
> Open **http://127.0.0.1:8000/** in your browser to access the chatbot.

---

## 📝 Usage Instructions
1️⃣ **Open Chatbot Interface**  
Navigate to `http://127.0.0.1:8000/` in your browser.

2️⃣ **Ask a Question**  
- For CDP-related queries, ask: *"How do I create a segment in Segment?"*  
- For general questions, ask: *"What is a Customer Data Platform?"*  

3️⃣ **Receive an Answer**  
- If the query is CDP-related, the chatbot extracts and returns relevant documentation.
- Otherwise, the chatbot provides an AI-generated response using **Google Gemini API**.

---

## 🤝 Contributing
We welcome contributions! Feel free to fork the repository and submit a **pull request** with improvements, bug fixes, or new features.

---

## 📌 Additional Resources
- **Project Live Demo**: [Live Link](#) *(Add live deployment link here)*
- **YouTube Video Explanation**: [Watch Video](#) *(Add YouTube tutorial link here)*

---

## 📄 License
This project is licensed under the **MIT License**.

---

### ✨ Created by **Siddharth Nama**
> Passionate about **AI, Web Development, and Data Science**.  
> Connect with me on **[LinkedIn](https://www.linkedin.com/in/siddharth-nama/) 

---

### ⭐ If you find this project helpful, consider giving it a **star** on GitHub! ⭐

