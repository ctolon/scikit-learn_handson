# scikit-learn_handson

Fun-Mooc Inria "Machine learning in Python with scikit-learn" kursuna ait özet-çözümler ile türkçe notlar içeren bir repository.


## İçindekiler

- [scikit-learn\_handson](#scikit-learn_handson)
  - [İçindekiler](#i̇çindekiler)
  - [Best Practices](#best-practices)
    - [Standart Scaler Ne Zaman Kullanılır?](#standart-scaler-ne-zaman-kullanılır)
      - [Teori](#teori)
  - [test\_train\_split vs cross\_validate](#test_train_split-vs-cross_validate)
  - [gridSearchCV vs RandomizedSearchCV](#gridsearchcv-vs-randomizedsearchcv)
    - [cross\_validate: cv parametresi için best practice değerler](#cross_validate-cv-parametresi-için-best-practice-değerler)
    - [Kategorik verileri analiz ederken hangi encoder seçilmeli?](#kategorik-verileri-analiz-ederken-hangi-encoder-seçilmeli)
      - [Teori](#teori-1)
  - [Genel Best Practiceler](#genel-best-practiceler)
  - [İsimlendirme Kuralları](#i̇simlendirme-kuralları)
  - [Önemli Notlar](#önemli-notlar)
  - [Modül İçerikleri](#modül-i̇çerikleri)


## Best Practices

### Standart Scaler Ne Zaman Kullanılır?

Özet: Doğrusal modellerde genelde StandartScaler() gerekir. Ağaç modellerinde gerekmez, uygulanırsa da sorun olmaz. Ayrıca StandartScaler nümerik verileri aynı uzaya aldığı için sadece **nümerik** samplelar için uygulanır.

#### Teori

Veri setinin özelliklerinin farklı aralıklara yayıldığını görüyoruz. Bazı algoritmalar, özellik dağılımları ile ilgili bazı varsayımlarda bulunur ve genellikle özelliklerin normalleştirilmesi, bu varsayımların ele alınmasına yardımcı olur.

Özellikleri ölçeklendirmenin bazı nedenleri şunlardır:

* Bir çift örnek arasındaki mesafeye dayanan modeller, örneğin k-en yakın komşular, her bir özelliğin mesafe hesaplamalarına yaklaşık olarak eşit katkıda bulunmasını sağlamak için normalleştirilmiş özellikler üzerinde eğitilmelidir.

* Lojistik regresyon gibi birçok model, optimum parametrelerini bulmak için sayısal bir çözücü (gradyan inişine dayalı) kullanır. Bu çözücü, özellikler ölçeklendiğinde daha hızlı birleşir.

* Bir makine öğrenimi modelinin özellikleri ölçeklendirmeyi gerektirip gerektirmediği, model ailesine bağlıdır. Lojistik regresyon gibi doğrusal modeller genellikle özelliklerin ölçeklendirilmesinden yararlanırken, karar ağaçları gibi diğer modeller bu tür bir ön işlemeye ihtiyaç duymaz (ancak bundan zarar görmez).

StandardScaler adlı bir scikit-learn dönüştürücü kullanarak böyle bir normalleştirmenin nasıl uygulanacağını gösteriyoruz. Bu dönüştürücü, her bir özelliği ayrı ayrı kaydırır ve ölçeklendirir, böylece hepsinde 0 ortalama ve bir birim standart sapma bulunur.

## test_train_split vs cross_validate

eğer veriseti 20binden fazla veri içeriyorsa genelde test_train_split kullanılır. Aksi takdirde cross_validate daha iyi bir seçim olabilir.

## gridSearchCV vs RandomizedSearchCV

eğer daha az hyper-parametreyi arayacaksak ve bunları hardcoded olarak arayacaksak, gridSearchCV seçilebilir. gridSearchCV yapısı itibariyle çok fazla hyper-parametre aramaya elverişli değildir, çok uzun süren bir arama yapar. RandomizedSearchCV ise hyper parametreler için aralıkları kendi belirler ve en iyiye yakın bir sonucu çok hızlı verir. Veri seti büyükse ve çok fazla hyperparametre tune edilecekse bu yöntem seçilebilir.

### cross_validate: cv parametresi için best practice değerler

Özellikle k-fold stratejisi için: cv=5 ya da cv=10

### Kategorik verileri analiz ederken hangi encoder seçilmeli?

Özet:  OrdinalEncoder Kategorik bir değişken herhangi bir anlamlı sıra bilgisi taşıyorsa (ordinal: örn. small, medium, large vs. gibi) o zaman kullanılır. Eğer sıra bilgisi yoksa (nominal: örn. male, female vs. gibi) o zaman one-hot encoder kullanılmalı.

#### Teori

Bir kodlama stratejisi seçmek, altta yatan modellere ve kategorilerin türüne (yani sıralıya karşı nominal) bağlı olacaktır.

Not

Genel olarak OneHotEncoder, aşağı akış modelleri doğrusal modeller olduğunda kullanılan kodlama stratejisidir, OrdinalEncoder ise genellikle ağaç tabanlı modellerde iyi bir stratejidir.

Bir OrdinalEncoder kullanmak, sıralı kategorilerin çıktısını verir. Bu, ortaya çıkan kategorilerde bir sıra olduğu anlamına gelir (örn. 0 < 1 < 2). Bu sıralama varsayımını ihlal etmenin etkisi, gerçekten aşağı yönlü modellere bağlıdır. Doğrusal modeller yanlış sıralanmış kategorilerden etkilenirken ağaç tabanlı modeller etkilenmeyecektir.

Yine de bir OrdinalEncoder'ı doğrusal modellerle kullanabilirsiniz, ancak şunlardan emin olmanız gerekir:

     orijinal kategorilerin (kodlamadan önce) bir sıralaması vardır;
     kodlanmış kategoriler, orijinal kategorilerle aynı sıralamayı takip eder. Bir sonraki alıştırma, OrdinalEncoder'ın doğrusal bir modelle kötüye kullanılması konusunu vurgulamaktadır.

Yüksek kardinaliteye sahip one-hot kodlama kategorik değişkenleri, ağaç tabanlı modellerde hesaplama verimsizliğine neden olabilir. Bu nedenle, orijinal kategorilerin belirli bir sırası olmasa bile, bu tür durumlarda OneHotEncoder kullanılması önerilmez.

## Genel Best Practiceler

* Preprocessing için fit ve transformu ayrı ayrı yapmak yerine fit_transform kullan.

* Feature tiplerini belirlemek için selector kullan (**ÖNCE HER ZAMAN MANUAL OLARAK CHECK ET**. Önemli notlarda açıklaması mevcut).

* Her zaman cross_validation ile skorları incele.

* Preprocessing ve modelleri tek tek yerine her zaman make_pipeline yada Pipeline, columntransform ile pipeline oluşturarak konfigüre et (temiz kod) .

* Eksik verileri ele almayı unutma.

* Encoder tipini seçerken best practicelere dikkat et, aksi takdirde dummy trape düşülür ve öğrenme başarısızlığa uğrar.

* Overfitting ve underfitting tespiti için her zaman validation_curve'e bakmak iyi bir seçimdir.

* Model başarısını arttırmak için automated hyper-tuning kullanmak önemlidir.


## İsimlendirme Kuralları

* High bias: underfitting, High Variance: overfitting olarak adlandırılır.

* Geleneksel olarak, Python'da alt çizgi değişkeni, ilgilenmediğimiz sonuçları depolamak için bir "çöp" değişkeni olarak kullanılır.

* Scikit-learn dökümanlarında veriler (data) genellikle X olarak adlandırılır ve hedef (target) genellikle y olarak adlandırılır.

* Bir modelin tahmini ile gerçek hedefleri karşılaştırarak elde edilen test puanı veya test hatasına atıfta bulunurken bir modelin genelleme performansına atıfta bulunulur. Genelleme performansı için eşdeğer terimler, tahmin performansı ve istatistiksel performanstır. Bir tahmine dayalı modeli eğitmenin hesaplama maliyetlerini değerlendirirken veya onu tahminlerde bulunmak için kullanırken, bir tahmine dayalı modelin hesaplama performansına atıfta bulunulur.

* scikit-learn kuralı: Verilerden bir öznitelik öğrenilirse, StandardScaler için mean_ ve scale_'de olduğu gibi, adı bir alt çizgi (ö.r. _) ile biter.

## Önemli Notlar

* Hedef değişkeniniz dengesizse (örneğin, bir hedef kategoriden diğerinden daha fazla örneğiniz varsa), makine öğrenimi modelinizi eğitmek ve değerlendirmek için özel tekniklere ihtiyacınız olabilir;

* Gereksiz (veya yüksek oranda ilişkili) sütunlara sahip olmak, bazı makine öğrenimi algoritmaları için sorun olabilir;
     
* Karar ağacının aksine, doğrusal modeller yalnızca doğrusal etkileşimleri yakalayabilir, bu nedenle verilerinizdeki doğrusal olmayan ilişkilerin farkında olun.

* Ölçeklendirilmemiş verilerle çalışmak (Ö.r. StandartScaler uygulanmamış), algoritmayı potansiyel olarak daha fazla yinelemeye zorlayacaktır. Ayrıca, gerekli yineleme sayısının tahmin edici (max_iter tarafından kontrol edilen) parametresi tarafından izin verilen maksimum yineleme sayısından daha fazla olduğu felaket senaryosu da vardır. Bu nedenle, max_iter değerini artırmadan önce verilerin iyi ölçeklendiğinden emin olun.

* Çapraz doğrulamanın (cross_validate) amacı, bir makine öğrenimi modelinin yeni verileri tahmin etme yeteneğini test etmektir. Ayrıca, aşırı uydurma veya seçim yanlılığı gibi sorunları işaretlemek için kullanılır ve modelin bağımsız bir veri kümesine nasıl genelleştirileceğine dair bilgiler verir.

* Stringlerin ve dolayısıyla kategorik özellikleri temsil etmek için nesne veri tipinin kullanıldığını biliyoruz. Bunun her zaman böyle olmadığını unutmayın. Bazen nesne veri türü, uygun şekilde biçimlendirilmemiş tarihler (dizeler) gibi diğer bilgi türlerini içerebilir ve yine de geçen süre miktarıyla ilgilidir. Daha genel bir senaryoda, make_column_selector'ı yanlış bir şekilde kullanmamak için veri çerçevenizin içeriğini manuel olarak incelemelisiniz.

* Eğer ampirik (train) error ile karşılaşırsanız yani modeli eğitimiz zaman train error 0 gelirse, o zaman shuffle split stratejisi uygulanabilir (bu problem en çok decision treelerde karşımıza çıkıyor).

## Modül İçerikleri

1-) Temel pandas ve görselleştirme, ölçeklendirme ve görselleştirme, cross_validation giriş, encodinge giriş, basit pipeline oluşturma

2-) cross_validation framework ayrıntılar ve görselleştirme, overfitting ve underfiting tespiti için validation_curve ve görselleştirme, shuffle splite giriş

3-) Hyper-parametre tuning (automated ve manual) ve görselleştirme


