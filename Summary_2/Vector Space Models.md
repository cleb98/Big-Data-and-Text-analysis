- # Vector Space Model
	- Il **Vector Space Model** (*VS*) è un modello per rappresentare **documenti come vettori**. È un sistema di retrieval semplice, ma efficace per il design di funzioni di ranking.
	  ![image.png](../assets/image_1683365189513_0.png){:height 170, :width 210}
	- Documenti e Queries sono rappresentati come dei vettori.
	  $$d_j = (w_{1,j},w_{2,j},...,w_{n,j}) \\ q = (w_{1,q},w_{2,q},...,w_{n,q})$$
	  Dove ogni dimensione corrisponde ad un _term_. La definizione di _term_ dipende dall'applicazione. Tipicamente i termini sono parole singole, keywords, o piccole frasi. Se i termini sono le parole del documento, la dimensione del vettore è pari alla cardinalità del documento $$|V|$$. Abbiamo quindi un iperspazio |V|-dimensionale.
		- > **_Esempio:_** 2 parole $$\rightarrow$$ spazio bidimensionale.
	- Se un termine compare nel documento, il suo valore nel vettore è non-zero. I componenti del vettore, chiamati anche _term weights_, possono essere calcolati in diversi modi.
	- ## Calcolo della similarità
	  Come già detto, si trasforma la query e il documento in un vettore e si calcola la similarità tra i due vettori. I documenti che sono meno distanti dalla query sono quelli più simili e quindi quelli con il ranking più alto.
		- > **_Oss:_** I vettori che descrivono i documenti e le query hanno le stesse dimensioni.
		- ### Similarità del coseno
			- ![image.png](../assets/image_1683369624907_0.png){:height 440, :width 247}
			- Solitamente si utilizza la *similarità del coseno*, nella quale si misura il coseno dell'angolo formato da i due vettori. Utilizzando il coseno la similarità tra il documento $$d_j$$ e $$q$$ può essere calcolata come:
			  $$cos(d_j,q) = \frac{d_j \cdot q}{||d_j|| \cdot ||q||} = \frac{\sum_{i=1}^N w_{i,j}w_{i,q}}{\sqrt{\sum_{i=1}^N w_{i,j}^2}\sqrt{\sum_{i=1}^N w_{i,q}^2}}$$
			- > **_Oss:_** Un valore del coseno pari a zero, sta ad indicare che documento e query sono ortogonali e non c'è match.
			  ![image.png](../assets/image_1683364953682_0.png){:height 212, :width 323}
			- > **_Oss:_** Possiamo concentrarci unicamente sul numeratore della formula del coseno.
			- > **_Esempio:_**
			  ![image.png](../assets/image_1683370433862_0.png)
			  **Problemi:**
			  + Il documento d4 dovrebbe essere più rilevante del documento d3, poiché menziona "presidential" solo una volta, mentre ind4 viene menzionato molte più volte;
			  + I documenti d2 e d3 hanno lo stesso score, ma d3 è più rilevante di d2 e dovrebbe avere uno score più alto.
	- ## Vector space Model +
		- Consideriamo anche la **term frequency (TF)**, cioè si contano quante si ripetono le parole nella query e nel documento.
		  $$TF(w,d) =count(w,d)$$
		- In questo modo i _term weights_ non sono più $$0$$ e $$1$$, ma corrispondono al numero di volte che i _terms_ compaiono nel documento.
	- ## Vector space Model ++
		- Consideriamo anche la **document frequency (DF)**, cioè si considera la frequenza della parola nella collezione dei documenti.
		  $$\frac{\text{documenti in cui compare la parola}}{\text{documenti totali}}$$
		- Le parole che sono più frequenti, sono poco utili, non discriminano a sufficienza.
			- > **_Oss:_** Solitamente, si considera la **document frequency inversa (IDF)** e si applica il logaritmo.
			  $$IDF(w) = log[\frac{M+1}{df(w)}]$$
			  dove, $$df(w)$$ è la document frequency di $$w$$, M è il numero totale di documenti nella collezione, e vience aggiunto $$1$$ per stabilità numerica.