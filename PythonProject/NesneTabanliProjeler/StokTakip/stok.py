import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sqlite3

class StokTakipUygulamasi():
    def __init__(self, root):
        # Ana pencereyi oluştur
        self.root = root
        self.root.title("Stok Takip Uygulaması")
        self.root.geometry("1200x800")  # Pencere boyutu

        # Veritabanı bağlantısını kur
        self.conn = sqlite3.connect("stok_takip.db")
        self.cursor = self.conn.cursor()

        # Eğer stok tablosu yoksa oluştur
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stok(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                urun_adi TEXT,
                adet INTEGER,
                birim_fiyati REAL,
                toplam_deger REAL
            )
        """)
        self.conn.commit()

        # Etiketler ve Giriş Alanları
        self.id_label = tk.Label(root, text="ID:", font=("Arial", 12))
        self.id_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.id_entry = tk.Entry(root, font=("Arial", 12))
        self.id_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.urun_adi_label = tk.Label(root, text="Ürün Adı:", font=("Arial", 12))
        self.urun_adi_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.urun_adi_entry = tk.Entry(root, font=("Arial", 12))
        self.urun_adi_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.adet_adi_label = tk.Label(root, text="Adet:", font=("Arial", 12))
        self.adet_adi_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.adet_adi_entry = tk.Entry(root, font=("Arial", 12))
        self.adet_adi_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        self.birimF_label = tk.Label(root, text="Birim Fiyatı:", font=("Arial", 12))
        self.birimF_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.birimF_entry = tk.Entry(root, font=("Arial", 12))
        self.birimF_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        self.toplamD_label = tk.Label(root, text="Toplam Değer:", font=("Arial", 12))
        self.toplamD_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.toplamD_entry = tk.Entry(root, font=("Arial", 12))
        self.toplamD_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # İşlem Butonları
        self.ekle_buton = tk.Button(root, text="Ekle", command=self.ekle, font=("Arial", 12), bg="green", fg="white")
        self.ekle_buton.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        self.duzelt_buton = tk.Button(root, text="Düzelt", command=self.duzelt, font=("Arial", 12), bg="orange", fg="white")
        self.duzelt_buton.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
        self.sil_buton = tk.Button(root, text="Sil", command=self.sil, font=("Arial", 12), bg="red", fg="white")
        self.sil_buton.grid(row=5, column=2, padx=10, pady=10, sticky="ew")
        self.temizle_buton = tk.Button(root, text="Temizle", command=self.girdileri_temizle, font=("Arial", 12), bg="blue", fg="white")
        self.temizle_buton.grid(row=5, column=3, padx=10, pady=10, sticky="ew")

        # Arama Çubuğu
        self.arama_label = tk.Label(root, text="Ara:", font=("Arial", 12))
        self.arama_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.arama_entry = tk.Entry(root, font=("Arial", 12))
        self.arama_entry.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.arama_entry.bind("<KeyRelease>", self.arama)

        # Tablo Oluştur
        self.tablo = ttk.Treeview(root, columns=("ID", "Ürün Adı", "Adet", "Birim Fiyatı", "Toplam Değer"), show="headings", height=8)
        self.tablo.heading("ID", text="ID")
        self.tablo.heading("Ürün Adı", text="Ürün Adı")
        self.tablo.heading("Adet", text="Adet")
        self.tablo.heading("Birim Fiyatı", text="Birim Fiyatı")
        self.tablo.heading("Toplam Değer", text="Toplam Değer")
        self.tablo.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        self.tablo.bind("<ButtonRelease-1>", self.satir_sec)
        self.verileri_yukle()

        # Grid Düzenini Ayarlama
        root.grid_rowconfigure(7, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_columnconfigure(3, weight=1)

    def ekle(self):
        urun_adi = self.urun_adi_entry.get()
        adet_str = self.adet_adi_entry.get()
        birim_fiyat_str = self.birimF_entry.get()

        if not urun_adi or not adet_str or not birim_fiyat_str:
            messagebox.showerror("Hata", "Tüm alanları doldurun.")
            return

        try:
            adet = int(adet_str)
            birim_fiyati = float(birim_fiyat_str)
            toplam_deger = adet * birim_fiyati

            self.cursor.execute(
                "INSERT INTO stok (urun_adi, adet, birim_fiyati, toplam_deger) VALUES (?, ?, ?, ?)",
                (urun_adi, adet, birim_fiyati, toplam_deger)
            )
            self.conn.commit()
            self.verileri_yukle()
            self.girdileri_temizle()
        except ValueError:
            messagebox.showerror("Hata", "Adet ve Birim Fiyatı sayısal olmalıdır.")

    def arama(self, event):
        arama_metni = self.arama_entry.get().lower()
        for item in self.tablo.get_children():
            values = self.tablo.item(item, "values")
            if arama_metni in str(values[0]).lower() or arama_metni in values[1] or arama_metni in str(values[2]).lower() or arama_metni in str(values[3]).lower() or arama_metni in str(values[4]).lower():
                self.tablo.selection_set(item)
                self.tablo.see(item)
            else:
                self.tablo.selection_remove(item)

    def satir_sec(self, event):
        secili = self.tablo.selection()
        if secili:
            item = self.tablo.item(secili)
            values = item["values"]
            self.id_entry.delete(0, tk.END)
            self.id_entry.insert(0, values[0])
            self.urun_adi_entry.delete(0, tk.END)
            self.urun_adi_entry.insert(0, values[1])
            self.adet_adi_entry.delete(0, tk.END)
            self.adet_adi_entry.insert(0, values[2])
            self.birimF_entry.delete(0, tk.END)
            self.birimF_entry.insert(0, values[3])
            self.toplamD_entry.delete(0, tk.END)
            self.toplamD_entry.insert(0, values[4])

    def duzelt(self):
        secili = self.tablo.selection()
        if secili:
            id = self.id_entry.get()
            urun_adi = self.urun_adi_entry.get()
            adet = int(self.adet_adi_entry.get())
            birim_fiyati = float(self.birimF_entry.get())
            toplam_deger = adet * birim_fiyati

            self.cursor.execute("UPDATE stok SET urun_adi=?, adet=?, birim_fiyati=?, toplam_deger=? WHERE id=?", (urun_adi, adet, birim_fiyati, toplam_deger, id))
            self.conn.commit()
            self.tablo.item(secili, values=(id, urun_adi, adet, birim_fiyati, toplam_deger))
            self.girdileri_temizle()

    def sil(self):
        secili = self.tablo.selection()
        if secili:
            id = self.tablo.item(secili)['values'][0]
            self.cursor.execute("DELETE FROM stok WHERE id=?", (id,))
            self.conn.commit()
            self.tablo.delete(secili)
            self.girdileri_temizle()

    def verileri_yukle(self):
        # Önce tabloyu temizle
        for i in self.tablo.get_children():
            self.tablo.delete(i)

        # Sonra veritabanından verileri yükle
        for row in self.cursor.execute("SELECT * FROM stok"):
            self.tablo.insert("", "end", values=row)

    def girdileri_temizle(self):
        self.id_entry.delete(0, tk.END)
        self.urun_adi_entry.delete(0, tk.END)
        self.adet_adi_entry.delete(0, tk.END)
        self.birimF_entry.delete(0, tk.END)
        self.toplamD_entry.delete(0, tk.END)

# Program doğrudan çalıştırıldığında burası devreye girer
if __name__ == "__main__":
    root = tk.Tk()  # Ana pencere
    app = StokTakipUygulamasi(root)  # Sınıftan nesne oluştur
    root.mainloop()  # Pencereyi sürekli açık tut
