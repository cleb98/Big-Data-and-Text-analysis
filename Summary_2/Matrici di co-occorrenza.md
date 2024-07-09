- # Matrici di co-occorrenza
	- Viene rappresentato quanto spesso una parola occorre in un documento (*Term-Document Matrix*), o quanto spesso una parola occorre insieme ad un altra parola (*Term-Term Matrix*).
	- ## Term-Document Matrix
		- Tabella, in cui ogni termine (parola) presente nei documenti viene rappresentato come una colonna e ogni documento viene rappresentato come una riga, o viceversa. Nella cella corrispondente al termine e al documento si indica la frequenza con cui il termine appare nel documento.
		- > **_Esempio:_** consideriamo tre documenti contenenti le seguenti frasi:
		  + Documento 1: "Il cane corre nel parco"
		  + Documento 2: "Il gatto dorme sul divano"
		  + Documento 3: "Il cane abbaia al gatto"
		  | Termine | Documento 1 | Documento 2 | Documento 3 |
		  | --- | --- | --- | --- |
		  | Il | 1 | 1 | 1 |
		  | cane | 1 | 0 | 1 |
		  | corre | 1 | 0 | 0 |
		  | nel | 1 | 0 | 0 |
		  | parco | 1 | 0 | 0 |
		  | gatto | 0 | 1 | 1 |
		  | dorme | 0 | 1 | 0 |
		  | sul | 0 | 1 | 0 |
		  | divano | 0 | 1 | 0 |
		  | abbaia | 0 | 0 | 1 |
		- <ins>Come trovare documenti simili?</ins>
			- Se le colonne che rappresentano due documenti hanno all'incirca gli stessi valori, probabilmente parlano delle stesse cose, sono caratterizzati dallo stesso contesto.
		- <ins>Come trovare parole simili?</ins>
			- Se le righe che rappresentano due parole hanno all'incirca gli stessi valori, probabilmente hanno lo stesso significato.
	- ## Term-Term Matrix
		- Molto spesso le colonne contengono i singoli termini piuttosto che i singoli documenti. In questo caso la matrice in questione ha una dimensione $$|V| \times |V|$$, dove $$V$$ è la dimensione del vocabolario.
		- Ogni cella della matrice registra il numero di volte in cui la parola della riga e la parola della colonna compaiono insieme in qualche contesto in un corpus di testo.
		- La maggior parte dei numeri nella matrice sono pari a zero, per cui viene utilizzata una rappresentazione vettoriale sparsa.
		- > **_Esempio:_**
		  + Frase 1: La mela è rossa.
		  + Frase 2: La banana è gialla.
		  |         | la | mela | è | rossa | banana | gialla |
		  |---------|----|------|---|-------|--------|--------|
		  | la      | 0  | 1    | 1 | 1     | 0      | 0      |
		  | mela    | 1  | 0    | 1 | 1     | 0      | 0      |
		  | è       | 1  | 1    | 0 | 1     | 1      | 1      |
		  | rossa   | 1  | 1    | 1 | 0     | 0      | 0      |
		  | banana  | 0  | 0    | 1 | 0     | 0      | 1      |
		  | gialla  | 0  | 0    | 1 | 0     | 1      | 0      |
	- Per determinare la similarità tra due termini, tra due documenti o tra termine e documento è possibile utilizzare la **[[Similarità del Coseno]]**.
	- > **_Problema_**: La frequenza grezza delle parole non è una grande misura di associazione tra le parole, in quanto è molto sbilanciata e parole come "il" e "di" sono molto frequenti, ma forse non sono le più discriminanti. A tal fine, si utilizzano metriche come la frequenza del termine * l'inverso della frequenza del documento (Term Frequency - Inverse Document Frequency *TF-IDF*) o l'informazione mutua positiva puntiforme (Positive Pointwise Mutual Information - *PPMI*).
	- ## TF - IDF
		- Combinazione di due fattori:
			- **Frequenza del termine (TF)**: indica la frequenza di una parola in un documento:
			  logseq.order-list-type:: number
			  $$tf(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
			  dove:
			  + $f_{t,d}$ è il conto grezzo di un termine $$t$$ all'interno di un documento $$d$$;
			  + $\frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$ è il numero totale di termini nel documento $$d$$;
			- **Inverso della frequenza del documento (IDF)**: assegna un peso maggiore alle parole più discriminanti
			  logseq.order-list-type:: number
			  
			  $$idf(t,D) = log \frac{N}{|\{ d \in D : t \in d\}|}$$
			  
			  dove:
			  + $N$ è il numero totale di documenti $$N = |D|$$
			  + $|\{ d \in D : t \in d\}|$ è il numero di documenti dove compare il termine
		- Per un termine all'interno di un documento vale la seguente relazione:
		  $$tfidf(t,d,D) = tf(t,d) \cdot idf(t,D)$$
		  Il tf-idf viene calcolato come il prodotto tra la frequenza del termine (tf) e l'inverso della frequenza del documento (idf). **Un peso elevato nel tf-idf viene raggiunto da una frequenza elevata del termine nel documento dato e da una bassa frequenza del documento del termine nell'intera collezione di documenti**. Pertanto, i pesi tendono a filtrare i termini comuni.
		- ## PPMI
			- Il Puntowise Mutual Information è una misura di quanto spesso due eventi x e y si verificano, rispetto a quello che ci aspetteremmo se fossero indipendenti.
			  $$pmi(x,y) = log_2 \frac{p(x,y)}{p(x)p(y)}$$
			- Questa misura può essere applicata ai vettori di co-occorrenza definendo l'associazione punto per punto tra una parola target w e una parola contestuale c.
			  $$pmi(w,c) = log_2 \frac{p(w,c)}{p(w)p(c)}$$
			- > **_Oss:_** Il PMI può assumere valori che vanno da - infinito a + infinito. Pertanto viene definito il PPMI, che porta i valori negativi a zero.
			  $$ppmi(w,c) = max\{log_2 \frac{p(w,c)}{p(w)p(c)},0\}$$
			- La matrice PPMI è una matrice di co-occorrenza modificata in cui ogni elemento $$ppmi_{ij}$$ rappresenta il valore PPMI della parola $$i$$ con il contesto $$j$$. La formula per il calcolo del valore PPMI è la seguente:
			  $$ppmi{ij} = max\{log_2 \frac{fij}{fi * fj}, 0\}$$
			  dove:
			  + $f_{ij}$ è il numero di volte in cui la parola i compare nel contesto j;
			  + $f_i$ è il numero di volte in cui la parola i appare in qualsiasi contesto;
			  + $fj$ è il numero di volte in cui il contesto j appare con qualsiasi parola.