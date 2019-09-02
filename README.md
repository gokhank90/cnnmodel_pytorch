# firstModel
Öncelikle projemizi çalıştırmak için interpreter ayarı yapmalıyız. Bunun için de 
Dockerfile'ımızı build etmeliyiz. 
```dockerfile
docker build -t mnist:pytorch -f /home/gkhnkrhmt/Desktop/docker/Dockerfile .
```
komutu ile Dockerfile build oluyor.Build ettiğimiz Dockerfile artık local makinemizde
bir Image haline geliyor . Geriye Image'ı run komutu ile çalıştırıp Container haline getirmek
kalıyor . 
```dockerfile
docker run -it -v `pwd`:/workspace mnist:pytorch 
```
komutu ile Container haline gelmiş oluyor . Projemizde artık "interpreter" olarak Docker sekmesinden
Dockerfile için verdiğimiz tag'i seçerek değişiklikleri onaylıyoruz. Böylece projemiz çalışmaya ,debug
edilmeye hazır . 

# Training
```
python train.py
```

# Test

```
python test.py
```

# TODO

- [ ] Split test and training code. 
- [ ] Write all the terminal commands. How to run codes ??

#Kod Anlatımı
Terminalden ;
```python
python3 cnnmodel.py
```
komutu ile programı çalıştırabilirsiniz.
 ##Modelin Eğitimi Nasıl Gerçekleşiyor?
 Başlangıçta modeli (**network**'ü) açıklayacak olursak ;
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        a = self.conv1.weight
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```
Modelimiz iki conv layer , 2 relu layer , 2 maxpool layer , 1 dropOut2d layer , 1 dropOut layer , 2 fullyconnected Layer
ve en son logaritmik softmax sınıflandırıcısı olduğu görülmektedir.
Öncelikle temel mantığın anlaşılması için model eğitiminin aşamalarından bahsedelim . **Net** class'ının **__init__** fonksiyonu 
model için declaration anlamına gelmektedir. forward kısmı ismi ise tanımlaması yapılmış fonksiyonların çağrıldığı aşamadır.
Modelimiz **x** parametresi yani **data**'yı parametre alıp ilk başta conv1 layer'ında işleme tabi tutar . 


Burada yapılan işlem ; Filtre sayısını 10 yap ve 28x28x1x64(MNİST veri setinden gelen data)'ü 5x5 filtre uygula . 
5x5 rastgele initialize edilmiş bu filtreler 28x28'lik matrisi tamamen döner. **Conv**'ın amacı resme filtre uygulayarak resmi
detaylandırmak böylece resim için **featureExtraction** işlemi yapılmış olur. Conv1'in çıkışnda 24x24x10x64 elde edilir.
24x24x10x64 maxpool2d işlemine girer . Burda yapılan işlem ise 24x24'matrisler için 2x2'lik bir matrisler ile maksimum değeri 
alınır ve sonuç olarak 12x12x10x64 elde edilir. Daha sonrasında relu işlemi ile linearizasyon bozulur ve eksili değerler 0'a 
yuvarlanır , geriye kalan değerler ise olduğu gibi kalır . 

Bu işlemden hemen sonra **Conv2 layer**'ı ile 5x5 filtre ve filtre sayısı olarak 20 verilir . Bu adımda yapılan şey ; 
12x12'lik 10 matris 5x5'lik 10 matrisli 20 filtreden geçirilmesi . Sonuç olarak 8x8x20x64 shape'inde bir veri elde edilir.
Elde edilen bu veride model ezberinin önüne geçmek için DropOut2d işlemi yapılır . Bu işleme göre bazı channel'lar sıfırlanır , 
yani etkisi yok edilir. Ama shape olarak bir değişim olmaz .  

**Maxpool2d** işlemi ile 2x2'lik matrislere bakılarak maksimum değerlerin içerdiği 4x4x20x64 elde edilir. Bu işlem sonrasında 
relu ile linearizasyon bozulur ve 0'ın altındaki değerler 0'a yuvarlanır , geriye kalan değerler aynı kalacaktır.

En son elde edilen 4x4x20x64 lük veri dizi işlemi ile **64x320**'ye çevirilir . Bunun anlamı 64 veri var ve her veri için 320
feature . Sonrasında bu **320** feature **fc1 layer**'ına giderek **50** birime bağlanarak 50  sınıfa ayrılır. 50 sınıfa 
ayrılan bu veri relu işlemi ile linearizasyon bozulur ve eksili değerler 0'a yuvarlanır , geriye kalan değerler 
aynı kalacaktır.

Relu işleminin hemen ardından Dropout ile (default değeri 0,5) default oranında  bazı ağırlıklara dikkat edilmez ve  modelin ezberi
bozulur. Hemen sonrasında **fc2 layer**'ına giderek yapılan 50 sınıflandırma 10 birimle tam bağlanarak 10 sınıfa ayrılır.
10 sınıf için logaritmik Softmax ile değerler üretir ve target'e en yakın olan değer seçilir . 

##Verilen Hiper Parametreler
n_epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
plot_interval = 0

##Verileri Çekme
```python
#Train Verisinin MNIST dataset'inden Çekilmesi , Tensor'a Transform etme , Data Load ...
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/gkhnkrhmt/datasets', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
```
Eğitim verisi = 60000
```python
#Test Verisinin MNIST dataset'inden Çekilmesi , Tensor'a Transform etme , Data Load ...
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/gkhnkrhmt/datasets', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)  # parametreleri burda verdik batch için
```
Test verisi = 10000

```python
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)  # 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one).
```
test_loader'dan çekilmiş datasetlerini examples'a kaydediyoruz ve indisleri de tutuyoruz . İkinci satıra bakacak olursak 
**batch_idx** değişkeni test_loader'ın indekslerini tutmaktadır , examples ise test_loader'ın değerlerini tutmaktadır.

```python
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
```
optimizer olarak Stohastic Gradient Descent kullanılmaktadır.

##Modelin Eğitim Aşaması ve TrainLoss Bulma 
```python
network.train()
```
ile modelin train için kullanılacağı belirtilir. 
```python
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data) #burası ile network'e datayı yollayarak datayı eğitiyor.
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```
optimizer.zero_grad ile gradyan değerleri sıfırlanır. <br/>
output = network(data) ile alınan data yukarıda anlatmış olduğum **Model Nasıl Eğitiliyor** başlığındaki aşamaların uygulanması.
output değerimiz softmax ile döndürülen tahmin değeri olacaktır . <br/>
loss = F.nll_loss(output,target) ile tahmin edilen output değeri ile hedef arasındaki loss'u bize döndürür. <br/>
loss.backward() işlemi ile her ağırlığın loss'a olan etkisi bulunur . <br/>
optimizer.step() işlemi ile ağırlıktan ağırlığa olan etkisi çıkarılır ve ağırlık güncellenir.

```python
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')
            train_losses.append(loss.item())
            plot_interval += 1
            print(plot_interval)
```
Her hatayı ekrana yazdırmamak adına böyle bir if koşulu konmuştur. Bildiğiniz üzere **batch_idx** test_loader'ın ID değerlerini
tutmaktaydı. Eğitim yaparken test ile ilgili veri kullanıyoruz diye düşünmeyin . Burada amaç 0,1,2,.. gibi sıralı indeksleri
barındıran bir dizi kullanmak . Her neyse esas konumuza dönecek olursak burada yapılan işlem her 10 hata döndürümünde yalnızca bir kere
hatayı ekrana bastırmaktır. train_counter dediğimiz dizide ilk elaman 0 olup her adımda diziye eleman olarak bir öncekinin 
640 fazlasını  ekleyecektir. <br/>
#OPTİMİZER.PTH , MODEL.PTH ?
Kaydetme işlemlerinin hemen ardından train_losses dizisine o an ki loss değeri eklenir. Bu şekilde 640 ilerleye ilerleye gideceği 
için 60000 test setini 94 adımda tamamlayacaktır.

```python
for epoch in range(1, n_epochs + 1):
```
bu işlemler her epoch için tekrarlanır .

##TestLoss Hesaplama 
```python
createLosstest = test(createLosstest)
```
ile test fonksiyonu çağırılır.
```python
    network.eval()
    test_loss = 0
    correct = 0
```
network.eval() ile test tüm layer'lara modelin test modunda olduğu belirtilir. Training mode'a göre
 yalnızca batchnorm ya da dropout layer'ları çalışacaktır.
 ```python
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
```
with torch.no_grad() ile gradient hesabı önüne geçilir. Sonrasında ;
data'da 28x28x1x1000 shape'inde dizi tutulur. Yani girdilerimiz 28x28x1'lik olduğuna göre bu girdiye göre alınan 1000 resim 
data değişkeninde tutulmaktadır. **target**'te ise 1000x1'lik shape vardır ve target her resmin 10 sınıfa ayrılmış 
halidir. <br/>
Daha sonrasında network'e data verilerek output'a aktarılır.

