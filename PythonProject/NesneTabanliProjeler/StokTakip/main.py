import tkinter as tk
from tkinter import messagebox
import sqlite3
from stok import StokTakipUygulamasi  # Stok takip sınıfını dışarıdan import ettik

class KullaniciKayitVeGiris:
    def __init__(self, root):
        self.root = root
        self.root.title("Kullanıcı Girişi ve Kaydı")
        self.root.geometry("360x260")  # Ekran boyutunu ayarla
        self.root.config(bg="#f2f2f2")  # Arka plan rengini belirle
        
        # Veritabanı bağlantısı
        self.conn = sqlite3.connect("stok_takip.db")
        self.cursor = self.conn.cursor()
        
        # Kullanıcılar tablosunu oluştur
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS kullanicilar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kullanici_adi TEXT UNIQUE,
                parola TEXT
            )
        """)
        self.conn.commit()

        # Kullanıcı giriş ekranı
        self.kullanici_adi_label = tk.Label(root, text="Kullanıcı Adı:", bg="#f2f2f2", font=("Arial", 12))
        self.kullanici_adi_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.kullanici_adi_entry = tk.Entry(root, font=("Arial", 12))
        self.kullanici_adi_entry.grid(row=0, column=1, padx=20, pady=10)

        self.parola_label = tk.Label(root, text="Parola:", bg="#f2f2f2", font=("Arial", 12))
        self.parola_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.parola_entry = tk.Entry(root, show="*", font=("Arial", 12))
        self.parola_entry.grid(row=1, column=1, padx=20, pady=10)

        self.giris_buton = tk.Button(root, text="Giriş Yap", command=self.giris_yap, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.giris_buton.grid(row=2, column=0, columnspan=2, padx=20, pady=20)

        self.kayit_buton = tk.Button(root, text="Kayıt Ol", command=self.kayit_ol, font=("Arial", 12), bg="#008CBA", fg="white")
        self.kayit_buton.grid(row=3, column=0, columnspan=2, padx=20, pady=10)

    def giris_yap(self):
        kullanici_adi = self.kullanici_adi_entry.get()
        parola = self.parola_entry.get()

        if not kullanici_adi or not parola:
            messagebox.showerror("Hata", "Lütfen kullanıcı adı ve parola girin.")
            return
        
        self.cursor.execute("SELECT * FROM kullanicilar WHERE kullanici_adi=? AND parola=?", (kullanici_adi, parola))
        user = self.cursor.fetchone()

        if user:
            messagebox.showinfo("Başarı", "Giriş başarılı!")
            self.root.destroy()  # Ana pencereyi kapat
            stok_root = tk.Tk()  # Stok ekranını aç
            StokTakipUygulamasi(stok_root)  # Stok takip ekranını başlat
            stok_root.mainloop()
        else:
            messagebox.showerror("Hata", "Kullanıcı adı veya parola yanlış.")

    def kayit_ol(self):
        kayit_root = tk.Tk()
        KullaniciKayit(kayit_root, self.conn, self.cursor)  # Kayıt ekranına conn ve cursor'ı gönder
        kayit_root.mainloop()

class KullaniciKayit:
    def __init__(self, root, conn, cursor):
        self.root = root
        self.root.title("Kayıt Ol")
        self.root.geometry("400x300")  # Ekran boyutunu ayarla
        self.root.config(bg="#f2f2f2")  # Arka plan rengini belirle
        
        # Veritabanı bağlantısını kaydediyoruz
        self.conn = conn
        self.cursor = cursor

        self.kullanici_adi_label = tk.Label(root, text="Kullanıcı Adı:", bg="#f2f2f2", font=("Arial", 12))
        self.kullanici_adi_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.kullanici_adi_entry = tk.Entry(root, font=("Arial", 12))
        self.kullanici_adi_entry.grid(row=0, column=1, padx=20, pady=10)

        self.parola_label = tk.Label(root, text="Parola:", bg="#f2f2f2", font=("Arial", 12))
        self.parola_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.parola_entry = tk.Entry(root, show="*", font=("Arial", 12))
        self.parola_entry.grid(row=1, column=1, padx=20, pady=10)

        self.parola_tekrar_label = tk.Label(root, text="Parolayı Tekrar Girin:", bg="#f2f2f2", font=("Arial", 12))
        self.parola_tekrar_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.parola_tekrar_entry = tk.Entry(root, show="*", font=("Arial", 12))
        self.parola_tekrar_entry.grid(row=2, column=1, padx=20, pady=10)

        self.kayit_buton = tk.Button(root, text="Kayıt Ol", command=self.kayit_et, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.kayit_buton.grid(row=3, column=0, columnspan=2, padx=20, pady=20)

    def kayit_et(self):
        kullanici_adi = self.kullanici_adi_entry.get()
        parola = self.parola_entry.get()
        parola_tekrar = self.parola_tekrar_entry.get()

        if parola != parola_tekrar:
            messagebox.showerror("Hata", "Parolalar uyuşmuyor!")
            return

        if not kullanici_adi or not parola:
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
            return
        
        # Veritabanına kullanıcı kaydını ekle
        try:
            self.cursor.execute("INSERT INTO kullanicilar (kullanici_adi, parola) VALUES (?, ?)", (kullanici_adi, parola))
            self.conn.commit()
            messagebox.showinfo("Başarı", "Kayıt başarılı! Giriş ekranına yönlendiriliyorsunuz.")
            self.root.destroy()
            main_root = tk.Tk()
            KullaniciKayitVeGiris(main_root)
            main_root.mainloop()
        except sqlite3.IntegrityError:
            messagebox.showerror("Hata", "Bu kullanıcı adı zaten alınmış.")

if __name__ == "__main__":
    root = tk.Tk()
    app = KullaniciKayitVeGiris(root)
    root.mainloop()
