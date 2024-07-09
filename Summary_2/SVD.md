- # Singular Value Decomposition
	- La Decomposizione ai Valori Singolari (SVD) è un metodo per approssimare un dataset N-dimensionale utilizzando un numero minore di dimensioni. Ciò viene fatto attraverso una rotazione degli assi in direzione della maggior varianza dei dati. In altre parole, la SVD trova le dimensioni più importanti di un dataset, cioè quelle lungo le quali i dati variano di più.
	- > **_Oss:_** Esistono molti metodi correlati, come la:
	  + [[PCA]]
	  + Factor Analysis
	- ## Definizione
		- Sia $$A$$ una matrice $$m \times n$$. A può essere scomposta, fattorizzata, in 3 matrici:
		  $$A = U \Sigma V^T$$
		  dove:
		  + $A$ è la matrice di dati di input, di dimensioni $m \times n$ 
		  + $U$ sono i vettori singolari sinistri, di dimensioni $m \times r$ 
		  + $Σ$ è la matrice diagonale $r \times r$ dei valori singolari 
		  + $V$ sono i vettori singolari destri, di dimensioni $n \times r$
	- ## SVD and term-context matrix
		- L'SVD può essere applicata alla matrice termine-contesto, che rappresenta le frequenze di ogni parola in un contesto specifico. L'SVD suddivide questa matrice in tre matrici quadrate: una matrice di parole $$W$$, una matrice di valori singolari $$S$$ e una matrice di contesti $$C^T$$.
			- Ogni riga della matrice $$W$$ rappresenta ancora una parola, ma le colonne rappresentano dimensioni latenti (e non contesti specifici), in cui i vettori di colonna sono ortogonali tra loro e ordinati per la quantità di varianza.
				- > **_Oss:_** $$W==U$$
			- Ogni riga della matrice $$C^T$$ rappresenta ora le nuove dimensioni latenti, mentre le colonne rappresentano i contesti specifici. I vettori di riga sono ortogonali tra loro.
	- ## Truncated SVD
		- ![Schematic-representation-for-singular-value-decomposition-SVD-analysis.png](../assets/Schematic-representation-for-singular-value-decomposition-SVD-analysis_1685192263953_0.png)
		- La versione "troncata" dell'SVD utilizza solo le prime k dimensioni delle tre matrici quadrate utilizzate per suddividere la matrice termine-contesto iniziale.
		- In questo modo la matrice $$W$$ ha dimensione $$|V| \times k$$, e ogni riga rappresenta una parola come un vettore denso k-dimensionale (embedding). Inoltre, la nuova matrice approssimata rappresenta l'informazione più importante della matrice originale, poiché le prime k dimensioni hanno la massima quantità di varianza nei dati.
		- > **_Oss:_** I valori singolari nella matrice $$\Sigma$$ rappresentano l'importanza relativa delle diverse dimensioni nella rappresentazione della matrice termine-contesto. I valori singolari più grandi rappresentano le dimensioni che contengono la maggior quantità di varianza nei dati e sono quindi le dimensioni più importanti per la rappresentazione della matrice termine-contesto.
		- > **_Oss:_** In genere, valori singolari e autovalori sono concetti distinti.