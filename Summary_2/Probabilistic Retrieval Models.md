- # Probabilisti Retrieval Models
	- I modelli di retrieval probabilistici si basano su una funzione che stima la probabilità che un documento sia rilevante rispetto alla query.
	- Introducono una variabile casuale binaria $R$ che rappresenta la rilevanza e modellano la query e i documenti come osservazioni di variabili casuali.
	  $$p(R = 1 | d, q)~~\text{dove}~~ R \in \{0,1\}$$
	  In altre parole, ci si chiede qual è la probabilità che il documento sia rilevante, data la query e il documento.
	- ## Definizione
		- Supponiamo di avere una tabella con:
		  + Query
		  + Documento restituito
		  + Valutazione dell’utente
		  Allora:
		  $$p = \frac{\text{numero di volte in cui il documento è stato utile}}{\text{numero di volte che il documento è stato restituito}}$$
		- > **_Oss:_** 
		  + Il sistema così definito impone di avere una rilevanza per ogni coppia query-documento. Quindi, un documento appena inserito nel sistema potrebbe non essere mai restituito.
		  + Il sistema non è gestibile per la sua mole. Esistono troppe combinazioni possibili di query e documenti.
		- ### Soluzione
			- È necessario introdurre un’approssimazione:
			  $$p(R = 1 | d, q) \approx p(q | d, R = 1)$$
			- Quindi, dato un documento rilevante ($R=1$), si sta cercando di calcolare qual è la probabilità che l’utente faccia la query $q$ (per restituire $d$). In altre parole, si è interessati ai casi in cui gli utenti hanno apprezzato un determinato documento e si vuole capire che tipo di query sono state utilizzate.
			- >**_Oss:_** viene fatta l’assunzione che l’utente formuli la query basandosi su un documento rilevante ma immaginario.
			- <ins>Come si calcola questa probabilità?</ins>
				- Vengono calcolate le probabilità condizionate di osservare la query per ogni documento che si suppone essere rilevante. Le probabilità vengono utilizzate per classificare i documenti (_document ranking_).
				- Il **modello n-gram** può essere utilizzato nell’ambito dei modelli di retrieval probabilistici per approssimare la probabilità condizionata di osservare una query dato un documento, stimando le frequenze degli n-grammi nel documento e utilizzando tali frequenze per determinare la probabilità della query.
	- # N-gram
		- **Obiettivo:** calcolare la probabilità di una parola data una frase, un documento…
		- **Possibile Soluzione:** calcolare la probabilità di una parola $w$ dato un contesto $h$ attraverso i conteggi di frequenza relativa. In altre parole si va a considerare il rapporto tra il conteggio delle occorrenze di una specifica parola o sequenza di parole rispetto al conteggio totale delle parole nel
		  contesto desiderato:
		  $$p(w|h) = \frac{\text{count}(h,w)}{\text{count}(h)}$$
		  Dove:
		  + $\text{count}(h,w)$ rappresenta il numero di volte in cui la parola $w$ appare nel contesto $h$
		  + $\text{count}(h)$ rappresenta il numero totale di parole presenti nel contesto $h$
		- >  **_Esempio:_**
		  + $w = \text{che}$
		  + $h = \text{la sua acqua è così trasparente}$
		  $$p(\text{che}|\text{la sua acqua è così trasparente}) = \frac{\text{la sua acqua è così trasparente che}}{\text{la sua acqua è così trasparente}}$$
		- > **_Oss:_**
		  + Dal punto di vista computazionale è estremamente oneroso, ci sono troppe combinazioni possibili;
		  + Si potrebbero perdere dei sinonimi, frasi simili ma leggermente diverse non vengono conteggiate.
		- ### Soluzione
			- Si assume che la probabilità della parola successiva alla frase considerata dipenda unicamente da una finestra di parole precedenti di dimensione fissata. Un modello bigram (2-gram) considera una parola precedente, un modello trigram ne considera due e, in generale, un modello n-gram considera n-1 parole del contesto precedente.
			  + _bigram:_
			  $$p(w_i | w_1w_2…w_{i-1}) \approx p(w_i|w_{i-1})$$
			  + _trigram:_
			  $$p(w_i | w_1w_2…w_{i-1}) \approx p(w_i|w_{i-1} w_{i-2})$$
			- >**_Esempio:_**
			  + $p(\text{che}|\text{la sua acqua è così trasparente})$ diventa 
			  + $p(\text{che}|\text{trasparente})$
		- Le probabilità possono essere stimate tramite la **maximum likelihood estimation (MLE)** che consiste nel contare le occorrenze nel corpus e normalizzare i valori.
		  + _bigram:_
		  $$ P(w_n|w_{n−1}) = \frac{\text{count}(w_{n−1},w_n)}{ \text{count}(w_{n−1})}$$
		  + _n-gram:_
		  $$ P(w_n|w_{n−N+1}^{n-1}) = \frac{\text{count}( w_{n−N+1}^{n-1},w_n)}{ \text{count}( w_{n−N+1}^{n-1})}$$
			- > **_Esempio:_**
			  + `<s> I am Sam </s>`
			  + `<s> Sam I am </s>`
			  + `<s> I do not like green eggs and ham </s>`
			  Ecco i calcoli per alcune probabilità di bigrammi da questo corpus:
			  + `p(am|I) = 2/3 = .67` 3 frasi contengono `I` ma solo 2 `I` vengono preceduti da `am`
			  + `p(I|<s>) = 2/3 = .67` 
			  + `p(Sam|<s>) = 1/3 = .33`
			  + `p(</s>|Sam) = 1/2 = .5`
			  + `p(Sam|am) = 1/2 = .5` 
			  + `p(do|I) = 1/3 = .33`
		- > **_Oss:_**
		  + `p(i|<s>) = 0.25`
		  + `p(english|want) = 0.0011`
		  + `p(food|english) = 0.5`
		  + `p(</s>|food) = 0.68`
		  Calcoliamo la probabilità  della frase ` I want English food` semplicemente moltiplicando le probabilità dei bigrammi come segue:
		  `P(<s> i want english food </s>) =`
		  `P(i|<s>)P(want|i)P(english|want)P(food|english)P(</s>|food)=.25×.33×.0011×0.5×0.68 = .000031`
		  Poiché le probabilità sono (per definizione) minori o uguali a 1, più probabilità si moltiplicano, più il prodotto diventa piccolo. Per questo motivo si calcolano le probabilità in formato logaritmico:
		  $$ log(p1 × p2 × p3 × p4) = log(p1) +log(p2) +log(p3) +log(p4)$$
		  In questo modo si evita l’underflow e la computazione è più veloce.
		- > **_Oss:_**
		  + Gli n-grammi svolgono un lavoro migliore se aumentiamo il valore di n;
		  + Gli n-grammi funzionano correttamente soltanto se il test corpus è simile al training corpus, ma nella realtà difficilmente ci si trova in questa situazione. Parole presenti nel set di addestramento ma non nel set di test potrebbero causare una moltiplicazione che porta a una probabilità totale nulla. Per questo motivo viene introdotto lo **smoothing**
		- ## Smoothing
			- Per evitare che il modello di linguaggio assegni una probabilità di zero a eventi non osservati, lo **smoothing** redistribuisce una piccola quantità di probabilità da eventi più frequenti a quelli mai visti.
				- ### Laplacian Smoothing
				  $$ P^*(w_n|w_{n−1}) = \frac{\text{count}(w_{n−1},w_n)+1}{ \text{count}(w_{n−1})+V}$$
				  In questo modo vengono eliminate le probabilità nulle.
			- > **_Oss:_** Lo smoothing potrebbe causare notevoli cambiamenti nei conteggi e, di conseguenza, nelle probabilità. Ciò potrebbe portare a una significativa riduzione delle probabilità che erano inizialmente relativamente elevate, poiché una parte considerevole della probabilità viene assegnata agli elementi con probabilità zero.
	- A questo punto, disponiamo di una **formula** che può essere utilizzata **per il reperimento e il ranking dei documenti**. Le probabilità sono fornite da un modello di linguaggio che non assegna una probabilità nulla a nessuna parola della query, anche nel caso in cui ci siano parole che non compaiono nei documenti.
	  $$q = w_1w_2…w_n$$
	  $$p(q|d) = p(w_1|d) \times … \times p(w_n|d)$$
	  $$ f(q,d) = log p(q|d) = \sum_{i=1}^n log p(w_i|d)= \sum_{w \in V}\text{count}(w,q)log p(w|d)$$
		- > **_Oss:_** Se una parola non è contenuta in un documento, la sua probabilità sarà proporzionale alla probabilità della parola nell’intera collezione di documenti.
		  $$p(w|d) =\begin{cases}p_{seen}(w|d) & \text{if }w\text{ is seen in } d \\ \alpha_dp(w|C) & \text{otherwise}\end{cases}$$
	- Con questa variazione è possibile riscrivere la funzione per il ranking come:
	  $$logp(q|d) =  \sum_{w \in V}\text{count}(w,q)log p(w|d) =  \\ \sum_{w \in V,~~c(w,d) > 0}\text{count}(w,q)log p_{seen}(w|d) + \sum_{w \in V,~~c(w,d) = 0}\text{count}(w,q)log \alpha_dp(w|C) = \\ \sum_{w_i \in d,~~ w_i \in q} \text{count}(w,q)[log \frac{p_{seen}(w_i|d)}{\alpha_dp(w_i|C)}] + nlog\alpha_d + \sum_{i=1}^Nlogp(w_i|C)$$