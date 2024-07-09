- # Search Engine Implementation
	- Un sistema di Information Retrieval consiste in 4 componenti:
		- **Tokenizer**:
			- Genera l’input del sistema di IR. In altre parole, estrae dal documento le caratteristiche fondamentali per confrontare i documenti tra loro (_feature generation_).
			- I documenti vengono rappresentati come vettori, in cui a ogni indice corrisponde una singola parola. Tuttavia possono essere adottate rappresentazioni alternative, come il conteggio delle occorrenze delle parole nel documento, o anche TF-IDF.
			- Il tokenizer più semplice suddivide il testo in parole in base agli spazi vuoti (_whitespace tokenizer_).
		- **Indexer**:
			- Gli indicizzatori dei motori di ricerca sono progettati per gestire una mole di dati molto più grande della memoria disponibile dal sistema.
			- Utilizzano sistemi di indicizzazione che caricano in memoria solo porzioni del corpus del documento per garantire un motore di ricerca rapido e funzionante.
			- Gli indici invertiti (_inverted index_) solo le principali struture dati utilizzate nei motori di ricerca. La struttura dati include:
				- Il _lexicon_, ovvero una tabella che contiene un elenco ordinato di tutti i termini distinti che compaiono nei documenti del corpus. Ogni termine è associato ad un identificatore univoco, noto come term ID. Inoltre, può contenere anche altre informazioni, come la frequenza dei termini e il numero di documenti in cui compaiono.
				- Il _postings file_ è un file o una struttura dati che memorizza le informazioni sulle occorrenze di un termine specifico nei  documenti indicizzati. La lista di postings può contenere ulteriori informazioni, come le posizioni specifiche in cui il termine appare in ciascun documento o altre statistiche associate a quel termine.
			- > **_Esempio:_**
			  + *Documento 1*: `Il cane corre veloce nel parco, il cane
			  salta felice`
			  + *Documento 2*: `Il gatto dorme tranquillo sul tappeto, il gatto miagola rumorosamente`
			  **Lexicon**:
			  | Termine | Documenti |Frequenza |
			  |------------|-----------|-----------|
			  | Il | 1,2 | 2 |
			  | cane | 1 | 2 |
			  | corre      | 1 | 1         |
			  | veloce     | 1         | 1         |
			  | nel        | 1         | 1         |
			  | parco      | 1         | 1         |
			  | salta      | 1         | 1         |
			  | felice     | 1         | 1         |
			  | gatto      | 2         | 2         |
			  | dorme      | 2         | 1         |
			  | tranquillo | 2        | 1         |
			  | sul        | 2         | 1         |
			  | tappeto    | 2         | 1         |
			  | miagola    | 2         | 1         |
			  | rumorosamente | 2 | 1         |
			  **Postings File**:
			  | Termine    | Documenti  |
			  |------------|------------------------------|
			  | Il         | 1,2                         |
			  | cane       | 1                            |
			  | corre      | 1                            |
			  | veloce     | 1                            |
			  | nel        | 1                            |
			  | parco      | 1                            |
			  | salta      | 1                            |
			  | felice     | 1                            |
			  | gatto      | 2                            |
			  | dorme      | 2                            |
			  | tranquillo | 2                            |
			  | sul        | 2                            |
			  | tappeto    | 2                            |
			  | miagola    | 2                            |
			  | rumorosamente | 2                         |
		- **Scorer/Ranker**:
			- L’indice invertito associa i termini ai documenti che li contengono.
			- I termini della query vengono valutati uno alla volta. Per ciascun termine, si recupera la lista dei documenti corrispondenti dall’indice invertito.
			- Per ogni termine della query, si calcolano i punteggi dei documenti basati sulle informazioni rilevanti (come frequenza del termine o probabilità del background). I punteggi vengono accumulati man mano.
			- I documenti vengono ordinati in base ai punteggi accumulati, e i migliori (solitamente i primi k) vengono restituiti come risultati della ricerca. Si evita di ordinare documenti con punteggio zero, risparmiando tempo
		- **Feedback/Learner**:
			- Utilizza il feedback dell’utente per migliorare il risultato della ricerca.