using NonlinearSolve, Plots, ProgressBars

function get_M_Delta(tj, x, bK; prev=[0.1, 0.1])
  function eqs(u, p)
    M = u[1]
    Delta = u[2]
    tj = p[1]
    x = p[2]
    bK = p[3]
    ress = [
      1-x - 2*cosh(M/tj)/(exp(Delta/tj - bK*(1-x)) +2*cosh(M/tj)),
      M - 2*sinh(M/tj)/(exp(Delta/tj - bK*(1-x)) +2*cosh(M/tj))
    ]
    return ress
  end
  prob = NonlinearProblem(eqs, prev, [tj, x, bK])
  root_solution = solve(prob, NewtonRaphson(), reltol=1e-6)
  return root_solution.u
end

function get_M_x(tj, Delta, bK; prev=[1, 1])
  function eqs(u, p)
    M = u[1]
    x = u[2]
    tj = p[1]
    Delta = p[2]
    bK = p[3]
    return [
      1-x - 2*cosh(M/tj)/(exp(Delta/tj - bK*(1-x)) +2*cosh(M/tj)),
      M - 2*sinh(M/tj)/(exp(Delta/tj - bK*(1-x)) +2*cosh(M/tj)),
    ] 
  end
  initial_guesses = prev
  prob = NonlinearProblem(eqs, initial_guesses, [tj, Delta, bK])
  root_solution = solve(prob, NewtonRaphson(), reltol=1e-15)
  return root_solution.sol
end


function get_arrays(;N=200)
  tj_vec = LinRange(0.01, 0.9, N) # T/J
  x_vec  = LinRange(0.01, 0.8, N)
  return [tj_vec, x_vec] 
end

# set ranges and K
# computational range:
function compute_parameters(;
    N=200, 
    K_over_J=0.0,
    num_threads=1,
    reset=true)
  print("\n============= defining vectors =============\n")
  tj_vec= get_arrays(N=N)[1]
  x_vec = get_arrays(N=N)[2]
  M      = zeros((N, N))
  Delta = zeros((N, N))
  print("\n============= start computing =============\n")
  # Define a worker function to fill the matrices
  for ix in ProgressBar(1:N)
      x = x_vec[ix]
      start_point = [0.5, 0.5]
      for (it, tj) in enumerate(tj_vec)
          bK = K_over_J * 1 / tj
          tmp = get_M_Delta(tj, x, bK, prev = start_point)
          M[it, ix]  = tmp[1] 
          Delta[it, ix] = tmp[2]
          start_point = [M[it, ix], Delta[it, ix]]
      end
  end

  return [M, Delta]
end

function lesgo()
  K_over_J_list = [0.0, 1.0, 2.0, 3.0]
  for K_over_J in K_over_J_list
    print("\n°°° computing for K/J =", K_over_J, " °°°\n")

    matrices = compute_parameters(;K_over_J=K_over_J, reset=true)
    p = heatmap(matrices[2])
    q = contour(matrices[2])
    display(q)
    # ORDERS
    # orders = compute_orders(K_over_J = K_over_J, reset=true, N=200)
    # plot_surface(orders[0], name="order_M.pdf")
    # plot_contour(orders[0], name="order_M.pdf")
    # plot_contour(orders[1], name="order_x.pdf")
  end
  return 0
end