
🚀 Multimodales KI-System für Marktanalysen
Ein flexibles, agentenbasiertes KI-Framework zur umfassenden Analyse von Finanzmarktdaten, das multimodale Eingaben wie Investorenberichte, Diagramme und Echtzeit-Nachrichten verarbeitet. Entwickelt im Rahmen eines Abschlussprojekts an der Hochschule Hannover.

🧩 Systemarchitektur
Das System besteht aus vier spezialisierten KI-Agenten, die zentral von einem Koordinator-Agenten gesteuert werden:

1. 🧠 Multimodaler RAG-Agent
Aufgabe: Verarbeitung und Analyse von IR-Dokumenten (z.B. PDFs) mittels Vektorsuche und Metadaten-Indexierung.
Technologien: LangChain, pdfplumber, ChromaDB, SentenceTransformers, Google Gemini
Features:

Extrahiert Text und Tabellen aus Dokumenten

Verknüpft Quellenangaben mit Firmenname, Jahr, Dokumenttyp und Seitenzahl

Speichert Dokumentsegmente für spätere Abfragen

2. 📈 Data Science & Prognose-Agent (DS-Agent)
Aufgabe: Analyse von Zeitreihen und Erstellung einfacher Finanzprognosen.
Technologien: LangChain, matplotlib, RegEx, Google Gemini
Features:

Extrahiert jährliche Finanzkennzahlen aus Texten

Führt Delta-basierte Extrapolationen durch

Generiert Prognosediagramme und speichert diese

3. 🌍 Echtzeit-Markt-Agent
Aufgabe: Bereitstellung aktueller Finanznachrichten, Aktienkurse und Sentiment-Analysen.
Technologien: yFinance, Alpha Vantage, NewsAPI, Tavily, HuggingFace, LangChain
Features:

Aggregiert Nachrichten von Yahoo Finance, CNBC, Reuters u.a.

Verfolgt Kursbewegungen und Handelsvolumen

Führt mehrsprachige Sentimentanalysen durch

Gibt Kauf-, Verkaufs- und Halteempfehlungen

4. 🧭 Koordinator-Agent
Aufgabe: Verwaltung und Orchestrierung der spezialisierten Agenten.
Technologien: LangChain, Gradio
Features:

Bietet eine einheitliche Benutzeroberfläche (Gradio)

Wählt passende Agenten basierend auf Nutzeranfragen aus

Generiert multimodale Antworten und verwaltet Fehler- und Fallbackszenarien

⚙️ Technologischer Stack
Programmiersprache: Python

Frameworks & Libraries: LangChain, Google Gemini, Gradio, Matplotlib, HuggingFace Transformers, pdfplumber, yFinance, ChromaDB

APIs: Google Generative AI, Tavily, Alpha Vantage, NewsAPI

🗂️ Verwendete Datenquellen
Unternehmen: Apple, Microsoft, Google (Alphabet), NVIDIA, Meta

Dokumenttypen: 10-K / 10-Q Reports, Earnings Call Transkripte, Präsentationsfolien

Zeitraum: 2020 – 2024

🧪 Projektübersicht
Woche 1:

Aufbereitung und Analyse von IR-Dokumenten (PDF)

Erstellung von Dokument-Embeddings mittels ChromaDB

Entwicklung eines Prognose-Tools

Woche 2:

Integration von Echtzeit-Finanzdaten und Sentimentanalyse

Aufbau des Koordinator-Agenten

Prototyp der Benutzeroberfläche mit Gradio

Fertigstellung der Agenten-Kollaboration und Logik

👥 Projektteam
Abschlussprojekt Hochschule Hannover
Betreuer: Hussam Alafandi

Studierende:

Alona Tkachenko

Natalia Musiienko

Mohamad Reza