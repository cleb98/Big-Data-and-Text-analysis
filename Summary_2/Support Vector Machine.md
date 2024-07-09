- # SVM #card
	- **SVM** è un algoritmo di apprendimento supervisionato utilizzato per **classificazione e regressione**. L’idea fondamentale alla base del funzionamento dell’SVM è trovare un **iperpiano di separazione** tra i dati, che **massimizzi la distanza tra i punti delle due classi e il piano stesso**. Per questo motivo SVM è un **classificatore lineare discriminativo definito a massimo margine**.
	- ## Margine
		- Se ci si trova in un piano euclideo, un problema di classificazione si risolve nel calcolare un'iperpiano $$w^T+b$$ che riesca a dividere correttamente i dati. Supponiamo di avere un iperpiano che divida perfettamente i dati; preso un generico punto $$x_i$$ del dataset:
		  + Se $$w^Tx_i+b > 0$$ allora l'elemento $$x_i$$ appartiene alla classe 0;
		  + Se $$w^Tx_i+b < 0$$ allora l'elemento $$x_i$$ appartiene alla classe 1;
		  Ci si chiede a questo punto quale può essere l'iperpiano migliore, ovvero quello che suddivide i dati nel modo migliore. **L'iperpiano migliore è quello che massimizza la distanza tra tutti i punti del dataset e se stesso**.
		- La **distanza di un generico punto** $$x_i$$ del dataset dall'iperpiano $$w^T+b$$ viene dalla formula:
		  $$r_i=\frac{|w^Tx_i+b|}{\sqrt{w_1^2+w_2^2+...+w_n^2}}=\frac{|w^Tx_i+b|}{||w||}$$
		- L'obiettivo è quindi quello di massimizzare la distanza punto - iperpiano per tutti i punti del dataset.
			- > **_Oss:_** Idealmente devo trovare la somma più grande di tutte queste distanze.
		- Non tutti i punti di dati sono ugualmente importanti ai fini della definizione dell'iperpiano. I punti più vicini all'iperpiano di separazione, chiamati anche **support vectors**, sono quelli che più fortemente ne influenzano la posizione.
		- **Riassumendo**:
		  + Si deve trovare un classificatore che classifichi correttamente tutti i punti del dataset;
		  + Si deve massimizzare la distanza di tutti i punti dal classificatore;
		- **Da un punto di vista matematico** ciò equivale a dire che:
		  + Dato un dataset $$D=\{(x_i,y_i), i=1,...,N\}$$ e $$y_i \in \{-1,1\}$$ e sia $$\rho$$ il margine non noto.
		  + Si vuole che tutti i punti del dataset siano classificati correttamente, e che idealmente distino $$\frac{\rho}{2}$$ dall'iperpiano, quindi: 
		  $$w^Tx_i+b \geq \frac{\rho}{2}~~\text{se}~~y_i=+1;$$
		  $$w^Tx_i+b \leq \frac{\rho}{2}~~\text{se}~~y_i=-1;$$
		  + Che può essere riassunto in un'unica equazione come:
		  $$y_i(w^Tx_i+b) \geq \frac{\rho}{2}~~\forall{(x_i,y_i) \in D};$$
		  + Per tutti i punti $$x_s$$ che si trovano sulla banda, esattamente a $$\frac{\rho}{2}$$ dall'iperpiano (quindi per i support vectors) la disuguaglianza diventa un'uguaglianza, quindi:
		  $$w^Tx_s+b = \frac{\rho}{2}$$
		  + Quindi per questi punti la distanza può essere riscritta come:
		  $$r_s=\frac{w^Tx_s+b}{||w||}= \frac{\rho}{2||w||}$$
		  + Poiché $$\rho$$ e $$w$$ non sono noti, è possibile dividere ogni componente di $$w$$ per $$\rho$$. In altre parole si sta assorbendo $$\frac{\rho}{2}$$ all'interno di $$w$$.
		  $$r_s=\frac{1}{||\hat{w}||}$$
		  + Quindi l'obiettivo della SVM è avere il margine $$\rho$$ pari a due volte la distanza $$r_s$$:
		  $$\rho=2r_s=\frac{2}{||\hat{w}||}$$
			- > **_Oss:_** Al posto di $$\hat{w}$$ posso usare semplicemente $$w$$.
	- ## SVM quadratic problem
		- Quindi il problema può essere riscritto come:
		  $$argmax_{w,b} \frac{2}{||w||}~~\text{s.t.}$$
		  $$y_i(w^Tx_i+b) \geq 1~~\forall{(x_i,y_i) \in D}$$
			- > Devo massimizzare il margine $$\rho$$ e contemporaneamente rispettare il vincolo di classificazione per tutti i punti del dataset.
			- > **_Oss:_** Concettualmente è come se avessi diviso a destra e a sinistra per $$\rho/2$$ e a sinistra ho assorbito $$\rho/2$$ dentro $$w$$.
		- Poiché **la norma di** $$w$$ **è sempre positiva**, **massimizzare una quantità, equivale a minimizzare l'inverso di quella quantità**. Inoltre è possibile considerare **il quadrato della norma poiché semplifica il problema di ottimizzazione**. Quindi il problema può essere riscritto come:
		  $$argmin_{w,b} ||w||^2~~\text{s.t.}$$
		  $$y_i(w^Tx_i+b) \geq 1~~\forall{(x_i,y_i) \in D}$$
		  Quindi, poiché la funzione obiettivo è quadratico e i vincoli sono lineari, il problema può essere formulato come un **problema di ottimizzazione quadratico QP**.
	- ## Ottimizzazione vincolata
		- Nell'SVM il **problema di ottimizzazione vincolata può essere risolto utilizzando il metodo dei moltiplicatori di Lagrange**. Nella formulazione generale, questo tipo di problemi comprende una funzione da minimizzare soggetta a dei vincoli positivi:
		  $$\text{funzione da ottimizzare}~~min~~f(x) \\ \text{soggetto ai vincoli}~~h_k(x) \geq 0~~\text{per}~~k \in 1,...,K$$
		- **Si incorporano i vincoli nella funzione da ottimizzare utilizzando i moltiplicatori di Lagrange**. In questo modo **si ottiene una funzione di Lagrange**. La forma generale della Lagrangiana in un problema di ottimizzazione con vincoli è data da:
		  $$L(x,\mu)=f(x)- \sum_i^K \alpha_i h_i(x) ~~ \text{con}~~\alpha_i \geq 0$$
		  dove:
			- $f(x)$ è la funzione che vogliamo minimizzare (**_Oss:_** per questo motivo consideriamo il segno negativo per i vincoli);
			- $\mu$ è l'insieme dei moltiplicatori di Lagrange. Si ha un moltiplicatore $$\alpha_i$$ per ciascun vincolo. I moltiplicatori devono essere maggiori o uguali a zero perché questo garantisce che i vincoli del problema originale siano soddisfatti.
		- A questo punto è possibile avere due condizioni:
		  1. Se per un dato $$x_i$$, $$h(x_i) > 0$$, allora il **vincolo è considerato inattivo** e il relativo **moltiplicatore di Lagrange**, $$\alpha$$, è impostato **a zero** $$\alpha=0$$. Ciò significa che il punto corrispondente non è sulla frontiera di decisione, quindi non è un support vector, e quindi non contribuisce alla soluzione. In questo caso, la soluzione è $$x^∗ |∇f (x) = 0$$, dove $$∇f (x)$$ è il gradiente della funzione obiettivo.
		  2. Se, invece, per un dato $$xi$$, $$h(xi) = 0$$, allora il vincolo è considerato attivo e il relativo moltiplicatore di Lagrange, $$α$$, è maggiore di zero. Ciò significa che il punto corrispondente si trova sulla frontiera di decisione, quindi è un support vector, e quindi contribuisce alla soluzione. In questo caso, la soluzione è $$x^∗ |∇f (x) − α^T ∇g(x) = 0$$, dove $$∇g(x)$$ è il gradiente del vincolo e $$α^T$$ è il trasposto di $$α$$.
	- ## KKT
		- Se mettiamo tutte queste condizioni insieme, **otteniamo le Karush-Kuhn-Tucker (KKT) condition**, ovvero un set di equazioni che sono necessarie ma non sufficienti per l'esistenza di un'unica soluzione ottimale $$x^*$$:
		  + $(1)~~∇f (x^∗) − α^T ∇h(x^∗ ) = 0 \Rightarrow$ il gradiente della funzione obiettivo è uguale al gradiente della funzione di vincolo. La soluzione ottimale deve rendere vera l'equazione.
		  + $(2)~~h(x^∗)≥0 \Rightarrow$ i vincoli devono essere soddisfatti per la soluzione ottimale.
		  + $(3)~~\alpha_i \geq 0 \Rightarrow$ i moltiplicatori di Lagrange non devono essere negativi.
		  + $(4)~~α^Th(x) = 0 \Rightarrow$ se un vincolo è attivo $$h(x) = 0$$, il moltiplicatore è positivo $$\alpha > 0$$. Se un vincolo è inattivo $$h(x) > 0$$, il moltiplicatore è nullo $$\alpha = 0$$. Il prodotto deve essere sempre uguale a zero.
			- > **_Oss:_** **Le KKT sono condizioni necessarie, ma potrebbero essere non sufficienti per trovare una soluzione ottimale**. Se la funzione obiettivo è convessa e la funzione di vincolo è differenziabile, allora le KKT sono sufficienti, quindi esiste la soluzione ottimale.
	- ## SVM: hard margin optimization
		- Nel caso dell' SVM la funzione obiettivo, quella da minimizzare è:
		  $$argmax_{w,b} ||w||^2$$
		  soggetta ai seguenti vincoli:
		  $$y_i(w^Tx_i+b) \geq 1~~\forall{(x_i,y_i) \in D}$$
		- I vincoli sono un insieme di equazioni lineari, che definiscono una funzione affine. Inoltre, **se le KKT sono soddisfatte**, significa che la **soluzione è globale** ed è **il minimo della funzione Lagrangiana**.
		- Il **minimo della funzione Lagrangiana** è la seguente:
		  $$\boxed{min_{w,b,\alpha}L(w,b,\alpha)=\frac{1}{2}||w||^2- \sum_{i=1}^N \alpha_i(y_i(w^Tx_i+b)-1)}$$
		  mentre le condizioni da dover soddisfare sono:
		  $$\alpha_i \geq 0; \\ y_i(w^Tx_i+b)-1 \geq 0; \\ \alpha_i[y_i(w^Tx_i+b)-1]=0$$
		  Quindi se trovo la soluzione del problema, ovvero il minimo della Lagrangiana, soddisferò anche le condizioni KKT.
			- > **_Oss:_** La prima equazione viene chiamata anche **primal form** (_formulazione primaria_).
		- Per trovare il **minimo della funzione Lagrangiana**, dobbiamo **derivarla e porre la derivata uguale a zero**. La Lagrangiana dipende da due parametri: $$w$$ e $$b$$
			- $$\frac{dL}{db}=0 \Rightarrow \sum_{i=1}^N \alpha_i y_i = 0$$
			  Devo bilanciare il contributo di vincoli che si trovano sopra e sotto l'iperpiano di separazione. Sto costruendo pertanto una banda intorno all'iperpiano che deve essere uniforme.
				- > **_Oss:_** stiamo considerando soltanto i punti per i quali i vincoli sono attivi, ovvero quelli per cui $$\alpha >0$$, ovvero i support vectors. Poiché $$\alpha$$ può essere soltanto positivo, per far sì che la sommatoria vada a zero è necessario considerare anche degli elementi con label -1. In altre parole, devo prendere lo stesso numero di vincoli sopra e sotto l'iperpiano.
			- $$\frac{dL}{dw}=0 \Rightarrow w=\sum_{i=1}^N \alpha_i y_ix_i$$
			  Mette in relazione i moltiplicatori di Lagrange con $$w$$, ossia il vettore che permette di calcolare la pendenza dell'iperpiano di separazione.
		- Sostituendo i risultati nella Lagrangiana si ottiene un'equazione che non dipende più da $$b$$ e da $$w$$, ma soltanto da $$\alpha$$:
		  $$\boxed{max_\alpha \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i^Tx_j)}$$
		  con vincoli:
		  $$\sum_{i=1}^N \alpha_i y_i = 0;~~ \alpha_i \geq 0~~i=1,..,N$$
			- > **_Oss:_** L'equazione prende il nome di **WOLFE dual formulation** (_formulazione duale_).
			- > **_Oss:_** Il duale è l'opposto del mio problema. Se per esempio voglio minimizzare una parabola (trovare il minimo della parabola), il duale richiede di massimizzare una parabola che ha la concavità opposta (trovare il massimo della parabola).
			  ![image.png](../assets/image_1673523792857_0.png){:height 211, :width 365}
			  Solo in una condizione la soluzione è uguale per tutti e due i problemi, ovvero il punto dove le due parabole si toccano e quindi dove sono verificate le KKT.
			- Quindi risolvere il problema di massimizzazione con un vincolo su $$\alpha$$, equivale a risolvere il problema originale. **Nella risoluzione dell'equazione duale possono verificarsi due condizioni**:
			  + $\alpha_i=0 \Rightarrow$ il punto $$x_i$$ e il vincolo associato non sono importanti. Non vengono utilizzati nel calcolo di $$w$$.
			  + $\alpha_i > 0 \Rightarrow$ il punto $$x_i$$ viene utilizzato nel calcolo di $$w$$. Tutti i punti che sono associati ad un moltiplicatore di Lagrange positivo sono i support vectors, per i quali il vincolo è attivo. Quindi **nel calcolo del margine vengono utilizzati unicamente i support vectors**.
	- ## Inferenza
		- Una volta allenato il classificatore e quindi aver trovato $$w$$ ed eventualmente anche $$b$$, per fare inferenza posso utilizzare sia la forma primaria che duale:
			- $\boxed{ \text{PRIMAL:}~~f(x)=w^Tx} \Rightarrow$ calcolo il segno della funzione per vedere dove si trova il nuovo elemento rispetto all'iperpiano e assegnare la classe associata.
			- $\boxed{ \text{DUAL:}~~f(x)=\sum_{i=1}^N\alpha_iy_ix_i^Tx} \Rightarrow$ devo salvare tutti i support vector $$x_i,\alpha_i,y_i$$ del dataset di training. Confronto la distanza di ogni support vector dal nuovo elemento e assegno la classe del support vector più vicino.
		- > **_Oss:_** le due formulazioni sono equivalenti, ma la seconda è molto più inefficiente in termini di memoria anche se è la più utilizzata nella pratica (perché è quella che viene utilizzata nel caso del kernel trick, nel caso di dataset non separabili linearmente).
- ## SVM: soft margin optimization #card
	- Se non è possibile trovare un'iperpiano in grado di separare perfettamente i dati, posso introdurre un'insieme di variabili $$\epsilon = \epsilon_1,\epsilon_2,...,\epsilon_N$$ (una per ogni punto del dataset) con $$\epsilon_i \geq 0$$, chiamate **slack variables**, le quali permettono di violare leggermente il vincolo sulla classificazione, che quindi può essere riscritta come:
	  $$y_i(w^Tx_i+b) \geq 1 - \epsilon_i~~\forall{(x_i,y_i) \in D}$$
	- Il problema di ottimizzazione diventa in questo caso:
	  $$min~~\frac{1}{2}||w||^2 + C \sum_i^N \epsilon_i \\ s.t.~~y_i(w^Tx_i+b) \geq 1 - \epsilon_i$$
		- > **_Oss:_** C è un iperparametro. Più C è basso è più si da importanza al margine (si cerca di trovare il margine più largo possibile, do maggiore importanza alle slack variables). Quindi si cerca un modello che generalizza. Mentre più C è alto e meno cerco di violare il vincolo di classificazione. Ciò equivale a trovare un modello più specializzato (do minore importanza alle slack variables).
- ## SVM non lineare #card
	- Se i dati non sono linearmente separabili non è possibile utilizzare le slack variables. **La soluzione in questi casi prevede di passare da uno spazio dove il problema non è lineare ad uno spazio dove il problema è lineare**.
	- Si cerca una funzione che mappa le coordinate dello spazio di partenza nelle coordinate del nuovo spazio: $$\phi: \phi \rightarrow \phi(x)$$.
		- > **_Oss:_** il nuovo spazio ha una dimensione maggiore dello spazio di origine.
	- Partendo dalla formulazione duale:
	  $$\sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i^Tx_j)~~~~f(x)=\sum_{i=1}^N\alpha_iy_ix_i^Tx$$
	- Poiché nella formulazione dell'SVM si ha $$x$$ che compare nel prodotto scalare, **non è necessario mappare esplicitamente** $$x$$ **nel nuovo spazio, ma cambiare solamente il modo in cui viene calcolato il prodotto scalare**. Noi sappiamo che uno spazio può essere definito da un insieme di coordinate, che vengono utilizzate per specificare la posizione dei punti all'interno dello spazio e dal prodotto interno (_inner product_) (nello spazio euclideo il prodotto interno coincide con il prodotto scalare), che viene utilizzato per definire la nozione di distanza o somiglianza tra i punti nello spazio. Quindi se cambio le distanze, cambio implicitamente anche lo spazio.
	- Questo concetto viene chiamato **Kernel Trick**, si definisce una **funzione K che equivale al prodotto scalare nel nuovo spazio**.
	- ## Kernel Trick
		- Utilizzando una funzione kernel $$K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$$ si può classificare senza bisogno di mappare i punti nel nuovo spazio:
		  $$f(x)= \sum_{i=1}^N \alpha_iy_iK(x_i,y_i)$$
		- Ciò viene garantito dal **teorema di Mercer**:
		  + Per ogni coppia di punti nello spazio originale scrivo la loro distanza nel nuovo spazio (nel nuovo spazio non conosco le coordinate ma conosco le distanze);
		  ![image.png](../assets/image_1673544034845_0.png){:height 143, :width 404} 
		  + Se la matrice è simmetrica e semidefinita positiva allora il determinante è $$\geq 0$$ e quindi si tratta di un kernel (matrice di Gram);
		  + Esiste quindi uno spazio dove gli elementi sono prodotti scalari.
		- ### Tipologie di kernel famosi
			- **Kernel lineare**: $$K(x_i,x_j)=x^Tx_i$$ non effettua nessuna trasformazione sui dati (funzione identità)
			- **Kernel polinomiale**: $$K(x_i,x_j)=(1+x^Tx_i)^p$$  mappo i dati in uno spazio con $$\binom{d+p}{d}$$ dimensioni (d è il numero di feature originale, p è il grado del polinomio)
			- **Kernel gaussiano**: $$K(x_i,x_j)= exp(- \frac{||x-x_i||^2}{2 \sigma^2})$$  mappo i dati in uno spazio con infinite dimensioni. Se ho infinite dimensioni trovo per forza un classificatore lineare che separa le classi. spiega intuitivamente come si ottengono le infinite dimensioni.
		- **Svantaggi del Kernel Trick**: devo calcolare il kernel per ogni punto del mio dataset. Questo è computazionalmente dispendioso.