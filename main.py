import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# CUDA kullanılabilir mi?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Donanım {device}")

# Dataset yükleniyor
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Test verileri yükleniyor
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# DataLoader ile verileri yüklüyoruz
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    X = X.to(device)  # X'i CUDA cihazına taşıyoruz
    y = y.to(device)  # y'yi CUDA cihazına taşıyoruz
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


class NeuralNetwork(nn.Module):
    # Modelimizi tanımlıyoruz
    def __init__(self):
        # Modelimizin katmanlarını tanımlıyoruz
        super().__init__()
        self.flatten = nn.Flatten() # Giriş katmanı
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ) # Çıktı katmanı

    def forward(self, x): # İleri besleme
        x = self.flatten(x) # Giriş katmanı
        logits = self.linear_relu_stack(x) # Çıktı katmanı
        return logits # Çıktı

model = NeuralNetwork().to(device)  # Modeli CUDA cihazına taşıyoruz
#print(model) # Modeli yazdırıyoruz

# Modeli eğitiyoruz
loss_fn = nn.CrossEntropyLoss() # Loss fonksiyonu
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # Optimizer


# Eğitim fonksiyonu
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # İleri besleme
        pred = model(X)
        loss = loss_fn(pred, y)

        # Geriye yayılım
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Her 100 batch'te loss'u yazdırıyoruz
        if batch % 100 == 0:
            # loss.item() loss'un değerini döndürür
            loss, current = loss.item(), (batch + 1) * len(X)
            # loss'u yazdırıyoruz
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



# Test fonksiyonu
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # Batch başına loss'u hesaplıyoruz
    num_batches = len(dataloader)
    # Modeli değerlendiriyoruz
    model.eval()
    # Loss ve doğruluk değerlerini sıfırlıyoruz
    test_loss, correct = 0, 0
    # Gradyan hesaplamalarını kapatıyoruz
    with torch.no_grad():
        # Test verilerini yüklüyoruz
        for X, y in dataloader:
            #CUDA cihazına taşıyoruz
            X, y = X.to(device), y.to(device)
            # İleri besleme
            pred = model(X)
            # Loss hesaplıyoruz
            test_loss += loss_fn(pred, y).item()
            # Doğru tahminleri sayıyoruz
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Loss'u yazdırıyoruz
    test_loss /= num_batches
    # Doğruluğu yazdırıyoruz
    correct /= size
    # Test verilerinin loss ve doğruluk değerlerini yazdırıyoruz
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# Test fonksiyonu
train(train_dataloader, model, loss_fn, optimizer)

# Modeli test ediyoruz
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Modeli kaydediyoruz
torch.save(model.state_dict(), "model.pth")
# Model Kayıt edildi
print("Saved PyTorch Model State to model.pth")


# Modeli yüklüyoruz
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

#
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
] # Sınıflar / Etiketler



model.eval() # Modeli değerlendiriyoruz
x, y = test_data[0][0], test_data[0][1] # Test verilerini yüklüyoruz
with torch.no_grad():# Gradyan hesaplamalarını kapatıyoruz
    x = x.to(device) # CUDA cihazına taşıyoruz
    pred = model(x) # İleri besleme
    predicted, actual = classes[pred[0].argmax(0)], classes[y] # Tahmin edilen ve gerçek değerleri alıyoruz
    print(f'Predicted: "{predicted}", Actual: "{actual}"') # Tahmin edilen ve gerçek değerleri yazdırıyoruz
    