# Implementation of C-RNN-GAN.



C-RNN-GANは、ランダムな入力からmidiを出力として学習していますが

入力を1次元配列にした上で外部から与える形に変更し、

出力も 数字の1次元配列に変更しました。



これにより

RNN-GANで seq2seq　の学習が可能となります。


