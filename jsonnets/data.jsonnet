// Config file containing all dataset information
local local_config = import "../local.jsonnet";
{
 local data_dir = local_config.data_dir, // all data live here

 // Fixed genre types
 local genre_list = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"],
 local null_list = ["all", "UNUSED_1", "UNUSED_2", "UNUSED_3", "UNUSED_4", "UNUSED_5", "UNUSED_6"],

 // Abstract dataset types
 local Dataset(train_path, dev_path, test_path, max_span_width, genre) = {
   train_path: data_dir + "/" + train_path,
   dev_path: data_dir + "/" + dev_path,
   test_path: data_dir + "/" + test_path,
   max_span_width: max_span_width,
   genres: if genre then genre_list else null_list,
 },

 local Dataless(max_span_width, genre) = {
   train_path: "",
   dev_path: "",
   test_path: "",
   max_span_width: max_span_width,
   genres: if genre then genre_list else null_list,
 },

 // OntoNotes English
 Ontonotes(segment, genre): Dataset("ontonotes/train.english." + segment + ".jsonlines",
                                    "ontonotes/dev.english." + segment + ".jsonlines",
				    "ontonotes/test.english." + segment + ".jsonlines",
			            30, genre),

 // Preco
 Preco: Dataset("preco/train.preco.512.jsonlines",
                "preco/train_dev.preco.512.jsonlines",
                "preco/dev.preco.512.jsonlines",
		15, false),

 // This ultimately unused config was for an experiment looking at dataset overlap
 Preco_data_fork(split): Dataset("preco/preco_fork/preco_" + split + ".fork.jsonlines",
 			 	 "preco/train_dev.preco.512.jsonlines",
 			 	 "preco/dev.preco.512.jsonlines",
				 15, false
				 ),

 // Litbank - the first config is a base version for debugging
 Litbank: Dataset("litbank/train.jsonlines",
                  "litbank/train.jsonlines",
                  "litbank/train.jsonlines",
		  20, false),

 Litbank_split(split): Dataset("litbank/train." + split + ".jsonlines",
                               "litbank/dev." + split + ".jsonlines",
                               "litbank/test." + split + ".jsonlines",
                               20, false),

 // QBcoref - the first config is a base version for debugging
 Qbcoref: Dataset("qbcoref/all_docs.512.jsonlines",
                  "qbcoref/all_docs.512.jsonlines",
		  "qbcoref/all_docs.512.jsonlines",
		  20, false),

 Qbcoref_split(split): Dataset("qbcoref/train." + split + ".jsonlines",
                               "qbcoref/dev." + split + ".jsonlines",
                               "qbcoref/test." + split + ".jsonlines",
                               20, false),

 // ARRAU
 Arrau: Dataset("arrau/train.512.jsonlines",
 		"arrau/dev.512.jsonlines",
 		"arrau/test.512.jsonlines",
 		15, false),

 // SARA
 Sara(split): Dataset("sara/train." + split + ".jsonlines",
 	              "sara/dev." + split + ".jsonlines",
                      "sara/test." + split + ".jsonlines",
		      10, false),

 // More multilingual configs

 // OntoNotes multilingual
 Ontonotes_ml(genre): Dataset("ontonotes_ml/train.ml.512.jsonlines",
 		              "ontonotes_ml/dev.ml.512.jsonlines",
			      "ontonotes_ml/test.ml.512.jsonlines",
			      30, genre),

 Ontonotes_ml_english(genre): Dataset("ontonotes_ml/english/train.english.512.jsonlines",
 		              "ontonotes_ml/english/dev.english.512.jsonlines",
			      "ontonotes_ml/english/test.english.512.jsonlines",
			      30, genre),


 Ontonotes_ml_arabic(genre): Dataset("ontonotes_ml/arabic/train.arabic.512.jsonlines",
 		              "ontonotes_ml/arabic/dev.arabic.512.jsonlines",
			      "ontonotes_ml/arabic/test.arabic.512.jsonlines",
			      30, genre),

 Ontonotes_ml_chinese(genre): Dataset("ontonotes_ml/chinese/train.chinese.512.jsonlines",
 		              "ontonotes_ml/chinese/dev.chinese.512.jsonlines",
			      "ontonotes_ml/chinese/test.chinese.512.jsonlines",
			      30, genre),


 // More multilingual configs below (some are missing a test set)

 //Semeval 2010 task 1
 Semeval: Dataset("semeval2010/train.512.jsonlines",
                  "semeval2010/dev.512.jsonlines",
                  "semeval2010/dev.512.jsonlines",
		  15, false),

 //Semeval 2010 task 1
 Semeval_es: Dataset("semeval/train.es.512.jsonlines",
                     "semeval/devel.es.512.jsonlines",
                     "semeval/test.es.512.jsonlines",
		     30, false),

 Semeval_ca: Dataset("semeval/train.ca.512.jsonlines",
                     "semeval/devel.ca.512.jsonlines",
                     "semeval/test.ca.512.jsonlines",
		     30, false),

 Semeval_nl: Dataset("semeval/train.nl.512.jsonlines",
                     "semeval/devel.nl.512.jsonlines",
                     "semeval/test.nl.512.jsonlines",
		     30, false),

 Semeval_it: Dataset("semeval/train.it.512.jsonlines",
                     "semeval/devel.it.512.jsonlines",
                     "semeval/test.it.512.jsonlines",
		     30, false),

 // For LOME: mixed semeval + ontonotes_ml + ru (no separate ru files)
 Mixed_ml: Dataset("multilingual/mixed.train.jsonlines",
                   "multilingual/mixed.dev.jsonlines",
                   "multilingual/mixed.dev.jsonlines",
		   15, false),

 Mixed_ml_dataless: Dataless(15, false),

}