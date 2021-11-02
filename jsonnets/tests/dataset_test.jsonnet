local data = import "../data.jsonnet";
{

  Ontonotes_512_false: data.Ontonotes(512, false),
  Ontonotes_512_true: data.Ontonotes(512, true),
  Ontonotes_test: data.Ontonotes(512, true) + data.Ontonotes_test(512),
  Ontonotes_to_litbank: data.Ontonotes(512, false) + data.Litbank_train,
}
