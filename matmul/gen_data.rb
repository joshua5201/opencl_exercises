require "time"
f = File.new("data.h", "w")
f.print("/* this file is automatically generated by gen_data.rb */\n")
rnd = Random.new(Time.now.to_i)
# change following numbers to produce matrix in different sizes
# matrix A (m x n) has m rows n columns
# matrix B (n x p) has n rows p columns
# matrix C (m x p) has m rows p columns
m = 1024
n = 1024
p = 1024

f.print("#define SIZE_M #{m}\n")
f.print("#define SIZE_N #{n}\n")
f.print("#define SIZE_P #{p}\n")
def gen_mat(m, n, name, f, rnd)
  f.print("float mat_#{name}[#{m*n}] = {\n")
  m.times do |i|
    n.times do |j|
      f.print("#{rnd.rand * 10}")
      if j != n - 1
        f.print(", ")
      end

    end
    if i != m - 1
      f.print(",\n")
    else
      f.print("\n")
    end
  end
  f.print("};\n")
end

gen_mat(m, n, "A", f, rnd)
gen_mat(n, p, "B", f, rnd)
