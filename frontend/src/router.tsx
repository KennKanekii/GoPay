import { createBrowserRouter } from 'react-router-dom'
import { AppLayout } from './ui/AppLayout'
import { Dashboard } from './views/Dashboard'
import { Home } from './views/Home'
import { Login } from './views/Login'
import { Signup } from './views/Signup'
import { SendMoney } from './views/SendMoney'
import { CreditScore } from './views/CreditScore'
import { NotFound } from './views/NotFound'
import { Privacy } from './views/Privacy'
import { Support } from './views/Support'
import { Terms } from './views/Terms'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'login', element: <Login /> },
      { path: 'signup', element: <Signup /> },
      { path: 'dashboard', element: <Dashboard /> },
      { path: 'send', element: <SendMoney /> },
      { path: 'credit', element: <CreditScore /> },
      { path: 'terms', element: <Terms /> },
      { path: 'privacy', element: <Privacy /> },
      { path: 'support', element: <Support /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])

