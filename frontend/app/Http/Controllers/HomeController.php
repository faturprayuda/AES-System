<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class HomeController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware(['auth', 'is_user']);
    }

    /**
     * Show the application dashboard.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index()
    {
        $data['titlePage'] = 'Automated Essay Scoring';
        return view('dashboard.user.index', $data);
    }

    public function checkPlagiarism()
    {
        $data['titlePage'] = 'Check Plagiarism';
        return view('dashboard.user.plagiarism', $data);
    }
}
