Per favorire la collaborazione tra i componenti del gruppo, il progetto è stato svolto con Google Colab. Qualora non si volesse utilizzare Google Colab, è necessario creare un ambiente con Python versione 3.7.13 ed installare le librerie necessarie, ovvero quelle presenti su Colab, tramite il file requirements.txt, generato dal comando pip freeze. 

Nella cartella Models, sono presenti molti più modelli di quelli scelti come migliori: questi, infatti, verranno poi selezionati automaticamente nei notebook di test e di deploy. 

In tutti i notebook, la funzione get_data precarica tutti i file salvati nella cartella Data/Prepared. Di conseguenza, può impiegare molti minuti. 
