- # Text Analysis
	- ## Text Data Access
		- Il collegamento con l'informazione può avvenire in due modi:
		  1. **pull**: l'utente si connette al sistema ed effettua delle ricerche di sua iniziativa, e si parla in questi casi di *querying* e *browsing*.
		  2. **push**: l'informazione viene fornita all'utente senza che si sia stata una richiesta specifica, si parla in questi casi di *sistemi di raccomandazione*.
	- > **_Oss:_**
	  **Search Engine vs Database:**
	  + *Search Engine*: operano con keywords che possono essere ambigue. La conoscenza del dominio dell'utente non è certa. È necessario ordinare per rilevanza i risultati. Ottenere documenti validi a partire da poche parole non è un task semplice.
	  + *Database*: l'utente sa cosa vuole ottenere, non c'è ambiguità nella richiesta. La difficoltà in questi casi ricade nell'ottimizzazione
	- ## Text Retrieval
		- ### Task
		  Data una collezione di documenti e un'esigenza informativa di un utente espressa sottoforma di keywords, si devono trovare i documenti che soddisfano la richiesta.
		- ### Problemi
		  + *Semantica*: cosa vuole esprimere l'utente con le sue keyword (le quali potrebbero essere ambigue)?
		  + *Semantica del/dei documenti*: come sapere quale documento restituire, qual è quello che soddisfa maggiormente la richiesta?
		  + *Metrica*: non esiste una metrica per valutare i risultati di un search engine poiché le valutazioni sono soggettive.
		- ### Ipotesi
		  + Abbiamo un set di parole indicizzate che fanno parte di un vocabolario:
		  $$\boxed{V = \{w_1,w_2,...,w_n\}}$$
		  + La query di un utente è un insieme di keyword, una sequenza di parole:
		  $$\boxed{Q = \{q_1,q_2,...,q_m | q_i \in V \}}$$
		  + Ogni documento è un insieme di parole:
		  $$\boxed{d_i = d_{i1}, ..., d_{is} | d_{ij} \in V} $$
		  + La dimensione della query è molto inferiore alla lunghezza del documento:
		  $$\boxed{|q_i| << |d_i|}$$
		  + Si ha una collezione di documenti:
		  $$\boxed{C = \{ d_1, ..., d_M\}}$$
		- ### Obiettivo
		  Trovare un sottoinsieme di $$C$$, chiamato $$R$$, funzione della query dell'utente:
		  $$\boxed{R(q) \subseteq C}$$
		  > **_Oss:_** Non si riuscirà mai a trovare i documenti veramente rilevanti per l'utente. Al massimo si raggiunge un'approssimazione di quello che l'utente sta cercando che quindi indichiamo con:
		  $$\boxed{R'(q)}$$
		  
		  **Come ottenere** $$R'(q)$$**?**
			- ## Document Selection:
			  $$
			  \begin{equation*}
			  f(q,d) =
			    \begin{cases}
			      1~~\Rightarrow \text{Documento d'interesse}\\
			      0~~\Rightarrow \text{Documento non d'interesse}
			    \end{cases}
			  \end{equation*}
			  $$
			  dove $$f(q,d)$$ è una funzione che, data la query, controlla ogni documento. Pertanto:
			  $$\boxed{R'(q,d) = \{d | f(q,d) = 1~~\forall d \in C\}}$$
			  > **_Oss:_** generalmente un search engine ritorna un **ranking**, poiché nella document selection si deve confrontare ogni documento con tutti gli altri.
			- ## Document Ranking
			  $f(q,d)$ ritorna un valore numerico che rappresenta il ranking del documento. Quindi:
			  $$R'(q)=\{d|f(q,d) \geq \theta\}$$
			  dove $$\theta$$ è un valore di soglia che abbiamo selezionato (ex. ritorna i primi 10 risultati, ritorna chi ha un ranking maggiore di tot).
			  **Come si ottiene** $$f(q,d)$$**?**
				- ## Tecniche basate sulla Similarità
					- ## [[Bag of Words (Bow)]]
					- ## [[Vector Space Models]]
				- ## Tecniche basate sulla Probabilità
					- ## [[Probabilistic Retrieval Models]]
				- > **_Oss:_** Dal 2018 (circa), queste tecniche vengono affiancate (e sostituite) dall'utilizzo di reti neurali.