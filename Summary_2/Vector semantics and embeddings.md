- # Vector semantics and embeddings
	- **Ipotesi distribuzionale**: il significato di una parola è legato al contesto in cui appare. In altri termini, parole che compaiono in contesti simili hanno spesso significati simili.
	- Le parole, le frasi, i documenti possono essere rappresentati come vettori numerici chiamati **embeddings** o **word embeddings** che ne rappresentano il significato all'interno del contesto in cui compaiono.
	- Esistono due tipi di rappresentazioni vettoriali:
		- **Rappresentazioni vettoriali sparse**: ogni parola (o documento) viene rappresentata come un vettore di dimensione pari al numero di parole nel corpus. La maggior parte dei degli elementi del vettori sono a zero poiché ogni parola è associata solo ad un sottoinsieme di parole nel corpus;
		  logseq.order-list-type:: number
			- **[[Matrici di co-occorrenza]]**;
			  logseq.order-list-type:: number
		- **Rappresentazioni vettoriali dense**: ogni parola (o documento) viene rappresentata come un vettore di dimensione fissata. Questi vettori sono densi perché ogni elemento ha un valore diverso da zero;
		  logseq.order-list-type:: number
			- *Decomposizione a valori singolari* (**[[SVD]]**) *e Analisi Semantica Latente* (**[[LSA]]**): vengono utilizzate delle tecniche di algebra lineare per ridurre la dimensionalità dei vettori;
			  logseq.order-list-type:: number
			- *Modelli ispirati a reti neurali* (**[[Word2vec]]**);
			  logseq.order-list-type:: number
			- *Embedding contestualizzati*: le rappresentazioni vettoriali dipendono dal contesto specifico;
			  logseq.order-list-type:: number
		- > **_Oss:_**
		  + I vettori densi sono più facili da utilizzare come feature nell'apprendimento automatico.
		  + I vettori densi possono generalizzare meglio rispetto alla memorizzazione di conteggi espliciti.
		  + I vettori densi possono catturare meglio la sinonimia e la similarità tra parole con significati simili.