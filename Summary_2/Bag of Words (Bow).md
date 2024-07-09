- È un modello in cui un testo (come una frase o un documento) viene rappresentato dal conteggio delle parole che appaiono in esso (si mantiene la molteplicità delle parole), e ignorando la grammatica e l'ordine delle parole.
- > **_Esempio:_** Se si hanno due documenti:
  1. `John likes to watch movies. Mary likes movies too.`
  2. `Mary also likes to watch football games.`
  I rispettivi BoW sono:
  `BoW1 = {"John":1,"likes":2,"to":1,"watch":1,"movies":2,"Mary":1,"too":1}`
  `BoW2 = {"Mary":1,"also":1,"likes":1,"to":1,"watch":1,"football":1,"games":1}`
- > **_Oss:_** BoW non considera le inflessioni. Per esempio "bello" e "bellissimo" sono elementi separati del vettore anche se hanno significati molto simili. Questa cosa non accade con il k-shingling se viene scelto un valore di k adeguato.
-
-
-
- DA RIVEDERE QUESTA PARTE!!
- *Data la query dell'utente, quali informazioni posso utilizzare?*
  1. **Term frequency:** tanto più la keyword compare nel documento e tanto più il documento è d'interesse;
  2. **Lunghezza documento:** più il documento è corto e meno è probabile che il documento contenga la keyword;
  3. **Frequenza della parola nel documento (Document Frequency):** bisogna associare un'importanza alle parole. Se la parola è presente molte volte nel documento, assume maggiore importanza;