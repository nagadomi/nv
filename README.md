# nv

libnvは俺用機械学習と画像処理、その他便利関数のライブラリです。

他のプロダクトで使っているので公開していますが、個人的な実験に使っているものなので、
バーションごとの互換性を保証する気はありませんし、細かいドキュメントを書く気もありません。
自分しか使ってないという前提で激しく変更するので、直接このライブラリを使わないほうがいいと思います。

ただライセンスの範囲で自由に使うことはできます。

## ノート

このライブラリはもともと[Imager::AnimeFace](http://anime.udp.jp/imager-animeface.html)のために作ったもので、Imager::AnimeFaceのnvsxというライブラリはnvの一部を切り出したものです。
ただnvxsはnvのかなり古いバージョンからforkしているので互換性がありません。

そのうちnvxsの`nv_face_detect`をこっちにリンクするように対応する予定ですが、
今のところImager::AnimeFaceのnvxsとヘッダーファイルがコンフリクトするので同じ環境では使えません。（別のディレクトリに入れてコンパイル時にパスを指定すれば可能）

## Installation

Ubuntu 12.04、FreeBSD 9.0 RELEASE、MinGW32、 Microsoft Visual C++ 2008 Express + Windows SDK for Windows Server 2008 and .NET Framework 3.5 で動作確認しています。

ここではUbuntu 12.04の場合を例にインストール方法を記述します。
他の環境は必要なライブラリをインストールして適切に配置すればビルドできます。

まず必要なパッケージをインストールします。

    sudo apt-get install libjpeg-dev libpng-dev libgif-dev libssl-dev

もし開発環境などを入れていない場合は、

    sudo apt-get install gcc g++ make autoconf automake libtool

などと入れておきます。

画像入力用のライブラリeiioをインストールします。

    wget https://github.com/nagadomi/eiio/tarball/master -O eiio.tar.gz
    tar -xzvf eiio.tar.gz
    cd nagadomi-eiio-*
    ./autogen.sh
    ./configure
    make
    sudo make install
    sudo ldconfig

あとはnvの最新を取ってきて、

    wget https://github.com/nagadomi/nv/tarball/master -O nv.tar.gz
    tar -xzvf nv.tar.gz
    cd nagadomi-nv-*
    ./autogen.sh
    ./configure
    make
    sudo make install
    sudo ldconfig

でインストールされます。

POPCNT命令に対応しているAMD CPUの場合は、configure に `--enable-popcnt` を付けるとビットベクトルのカウント処理が高速になります。Intel CPUの場合はSSE4.2に対応していれば自動で有効になります。

デフォルトでOpenMPとSIMD(AVXやSSEなど)が有効になります。
外したい場合は、`--disable-openmp` `--disable-native` などで外せます。

libopensslはlibcryptoの高速なSHA1の実装のために使っていますが、configure に`--disable-openssl`を付けると独自のSHA1実装を使うようになるので、OpenSSLとリンクする必要はなくなります。

    make check

一部機能のテストが走ります。乱数に影響を受ける部分があるのでまれに通らないことがあるかもしれません。

    sudo make uninstall

でアンインストールされます。

## プロセッサの対応について

可能な場合はSIMD(SSE,AVX)とOpenMPで最適化しています。
C言語版も用意してあるので、ARMなどのプロセッサでも動くと思いますが、かなり遅くなると思います(検証はしていない）。

nv_cudaには一部関数のCUDA実装がありますが、今はちょっとメンテされてないです。（そのうちします）

## 使用しているライブラリ

  - [tinymt32](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/)
  - [eiio](https://github.com/nagadomi/eiio) (optional)
  - [libcrypto](http://www.openssl.org/) (optional)

## 基本的な操作

読むハメになってしまったの場合のメモ。

全体的に`nv_matrix_t`という型で処理しています。名前はmatrixですが、多くは行列としては使っていなくて、主にベクトルの配列として使っています。

これはfloat型を要素に持つn次元のベクトルm個を操作できて、mはさらにrows、colsという2次元の添字でもアクセスできるので、ベクトルの2次元配列まで扱えます。（ベクトル自体が１次元配列なので3次元配列まで扱えます）

すでに意味不ですが、まずベクトルの1次元配列として扱う場合は、

    // 16次元のベクトル100個を確保
    int i, j;
    nv_matrix_t *vec_array = nv_matrix_alloc(16, 100);
    
    // 各ベクトルについて
    for (j = 0; j < vec_array->m; ++j) {
        // ベクトルの各要素について
        for (i = 0; i < vec_array->n; ++i) {
            // 0-1.0の一様乱数を設定
            NV_MAT_V(vec_array, j, i) = nv_rand(); 
        }
    }
    
とアクセスできます。`NV_MAT_V`は参照と代入どちらもできます。

次に2次元配列として扱う場合は、

    // 16次元のベクトル10x10個を確保
    int i, j, l;
    nv_matrix_t *vec_array = nv_matrix3d_alloc(16, 10, 10);
    
    // 縦列の各ベクトルについて
    for (l = 0; l < vec_array->rows; ++l) {
        // 横列の各ベクトルについて    
        for (j = 0; j < vec_array->cols; ++j) {
            // ベクトルの各要素について
            for (i = 0; i < vec_array->n; ++i) {
                // 0-1.0の一様乱数を設定
                NV_MAT3D_V(vec_array, l, j, i) = nv_rand(); 
            }
        }
    }

となります。このとき`m = rows * cols`となっているので、1次元の場合と同じように

    for (j = 0; j < vec_array->m; ++j) {
        // ベクトルの各要素について
        for (i = 0; i < vec_array->n; ++i) {
            // 0-1.0の一様乱数を設定
            NV_MAT_V(vec_array, j, i) = nv_rand(); 
        }
    }

ともアクセスできます。
逆に`nv_matrix_alloc`でベクトルの1次元配列を確保した場合は、`rows = 1; cols = m`となっています。

(row, col)の添字に対するmの添字は、

    NV_MAT_M(mat, row, col)

で求められます。

ベクトルの2次元配列は、主に画像を扱うために使っています。画像はBGR順の3次元のベクトルをcols個 * rows個並べたベクトルの2次元配列です。位置関係を考慮しない場合は、3次元のベクトル `rows*cols`個として処理できます。

要素がfloatなのは、SSEで4つ(AVXだと8個)同時に計算するためと、CUDAでパフォーマンスを出すためです（昔のCUDAは単精度しか速く出来なかった）。何次元の場合でも各要素はひとつの連続したメモリ上に確保されていて、_4次元以上_のベクトルの場合は、各ベクトルの先頭アドレスが8の倍数でアライメントされています。

### よくある処理

    nv_matrix_zero(mat);

全体を0で埋めます。

    nv_vector_zero(mat, j);

j番目のベクトルを0で埋めます。

    nv_vector_copy(mat1, j1, mat2, j2);

mat2のj2番目のベクトルをmat1のj1番目のベクトルにコピーします。

    nv_vector_normalize_L2(mat, j);

j番目のベクトルを破壊的にL2正則化します。

    float norm = nv_vector_norm(mat, j);

j番目のベクトルのL2ノルムを求めます。

    float dot = nv_vector_dot(mat1, j1, mat2, j2);

mat1のj1番目のベクトルとmat2のj2番目のベクトルの内積(ドット積)を求めます。

### 計算スレッド数の設定

環境変数 `OMP_NUM_THREADS` でスレッド数を指定できます。

    export OMP_NUM_THREADS=8
    make check
    
    export OMP_NUM_THREADS=1
    make check

### OpenCVのIplImageとの相互変換

configureに`--enable-opencv`オプションを付けると`IplImage`と`nv_matrix_t`の相互変換関数がビルドされます。

    IplImage *opencv_mat = nv_conv_nv2ipl(nv_mat);
    nv_matrix_t *nv_mat = nv_conv_ipl2nv(opencv_mat);

### 他

[src/tests](https://github.com/nagadomi/nv/tree/master/src/tests)の下に一部の大きめの処理のテストがあります。テストですが、使用サンプルみたいなものです。
