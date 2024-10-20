import { Component } from '@angular/core';
import { Router, Routes } from '@angular/router';

@Component({
  selector: 'app-catalogue',
  templateUrl: './catalogue.component.html',
  styleUrls: ['./catalogue.component.scss']
})
export class CatalogueComponent {
  routes: Routes;
  devMode: boolean = true;

  constructor(private router: Router) {
    this.routes = this.router.config
  }
}
