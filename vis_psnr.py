import matplotlib.pyplot as plt

psnr = []
with open('train_log.txt') as log:
    for line in log:
        if "Average PSNR" in line:
            psnr.append(float(line[-8:-3]))

plt.plot(psnr)
plt.xlabel('Epochs * 10')
plt.ylabel('PSNR')
plt.title('Validation PSNR every 10 epochs')
plt.savefig('validation_psnr.png')
